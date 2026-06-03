import json
import math
import os
import os.path
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data_utl


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


def load_rgb_frames(image_dir, vid, start, num):
    # (Fungsi ini biasanya untuk dataset berupa kumpulan gambar .jpg, bukan .mp4)
    frames = []
    for i in range(start, start + num):
        try:
            img = cv2.imread(os.path.join(image_dir, vid, "image_" + str(i).zfill(5) + '.jpg'))[:, :, [2, 1, 0]]
        except:
            print(os.path.join(image_dir, vid, str(i).zfill(6) + '.jpg'))
            continue
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_rgb_frames_from_video(vid_root, vid, start, num, label_folder=None, resize=(256, 256)):
    # MODIFIKASI: Mendukung sub-folder kelas SIBI
    if label_folder:
        video_path = os.path.join(vid_root, label_folder, vid + '.mp4')
    else:
        video_path = os.path.join(vid_root, vid + '.mp4')

    vidcap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(min(num, int(total_frames - start))):
        success, img = vidcap.read()
        if not success:
            break

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        if w > 256 or h > 256:
            img = cv2.resize(img, (256, 256))

        img = (img / 255.) * 2 - 1
        frames.append(img)

    vidcap.release()
    return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    count_skipping = 0
    vid_root = root['word']

    for vid in data.keys():
        # Filter subset
        if split == 'train':
            if data[vid]['subset'] not in ['train', 'val']:
                continue
        else:
            if data[vid]['subset'] != 'test':
                continue

        # MODIFIKASI: Cek path video di sub-folder
        label_folder = data[vid].get('folder', None)
        if label_folder:
            video_path = os.path.join(vid_root, label_folder, vid + '.mp4')
        else:
            video_path = os.path.join(vid_root, vid + '.mp4')

        if not os.path.exists(video_path):
            # print(f"File not found: {video_path}") # Debug jika perlu
            continue

        # Mengambil info durasi
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if mode == 'flow':
            num_frames = num_frames // 2

        if num_frames < 9:
            count_skipping += 1
            continue

        label = np.zeros((num_classes, num_frames), np.float32)
        # SIBI format: data[vid]['label'] adalah integer index kelas
        # Di WLASL ori, label disimpan di 'action' index 0
        try:
            class_idx = data[vid]['label']
        except KeyError:
            class_idx = data[vid]['action'][0]

        for l in range(num_frames):
            label[class_idx][l] = 1

        # Tambahkan info label_folder ke dalam tuple dataset
        dataset.append((vid, label, label_folder, 0, num_frames))

    print(f"--- Dataset {split} Loaded ---")
    print(f"Skipped videos (too short): {count_skipping}")
    print(f"Total instances: {len(dataset)}")
    return dataset


def get_num_class(split_file):
    with open(split_file, 'r') as f:
        content = json.load(f)
    
    classes = set()
    for vid in content.keys():
        try:
            class_id = content[vid]['label']
        except KeyError:
            class_id = content[vid]['action'][0]
        classes.add(class_id)
    return max(classes) + 1 # Mengembalikan total range kelas


class NSLT(data_utl.Dataset):
    def __init__(self, split_file, split, root, mode, transforms=None):
        self.split_file = split_file
        self.num_classes = get_num_class(split_file)
        self.data = make_dataset(split_file, split, root, mode, num_classes=self.num_classes)
        self.transforms = transforms
        self.mode = mode
        self.root = root

    def __getitem__(self, index):
        vid, label, label_folder, start_frame, nf = self.data[index]
        total_frames = 64

        # Random sampling window
        if nf > total_frames:
            start_f = random.randint(0, nf - total_frames) + start_frame
        else:
            start_f = start_frame

        # MODIFIKASI: Kirim info label_folder ke loader video
        imgs = load_rgb_frames_from_video(self.root['word'], vid, start_f, total_frames, label_folder=label_folder)

        imgs, label = self.pad(imgs, label, total_frames)

        if self.transforms:
            imgs = self.transforms(imgs)

        ret_lab = torch.from_numpy(label)
        ret_img = video_to_tensor(imgs)

        return ret_img, ret_lab, vid

    def __len__(self):
        return len(self.data)

    def pad(self, imgs, label, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]
            if num_padding > 0:
                # Padding dengan frame terakhir (freeze frame)
                pad_img = imgs[-1]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
            else:
                padded_imgs = imgs
        else:
            padded_imgs = imgs[:total_frames]

        # Menyesuaikan label agar panjangnya sesuai total_frames (T)
        # Label asli: (num_classes, num_frames) -> Target: (num_classes, total_frames)
        label_single = label[:, 0]
        label_padded = np.tile(label_single, (total_frames, 1)).transpose((1, 0))

        return padded_imgs, label_padded