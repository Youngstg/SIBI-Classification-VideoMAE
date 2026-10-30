import cv2
import os

def sample_video_frames(video_path, output_dir, num_samples=4):
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    # Extract filename without extension
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"Video {video_path} has 0 frames.")
        return

    # Calculate frame indices to sample (focus on the middle section where movement usually happens)
    # We will sample between 20% and 80% of the video duration.
    start_frame = int(total_frames * 0.20)
    end_frame = int(total_frames * 0.80)
    
    if num_samples > 1:
        step = (end_frame - start_frame) / (num_samples - 1)
        frame_indices = [int(start_frame + i * step) for i in range(num_samples)]
    else:
        frame_indices = [int(total_frames / 2)]
    
    print(f"Processing: {video_path}")
    print(f"Total frames: {total_frames}, Sampling at indices: {frame_indices}")

    for i, frame_idx in enumerate(frame_indices):
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            output_path = os.path.join(output_dir, f"{base_name}_sample{i+1}.png")
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_path}")
        else:
            print(f"Failed to read frame at index {frame_idx}")
            
    cap.release()

if __name__ == "__main__":
    # Define paths
    base_dir = r"D:\TUGASAKHIR\SourceCode"
    output_dir = os.path.join(base_dir, "sampel")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Videos to process
    videos = [
        os.path.join(base_dir, "folds_data", "videos", "Suka", "Suka_090.mp4"),
        os.path.join(base_dir, "folds_data", "videos", "Mau", "Mau_099.mp4")
    ]
    
    for video in videos:
        sample_video_frames(video, output_dir, num_samples=4)
        
    print("Proses pengambilan sampel gambar selesai!")
