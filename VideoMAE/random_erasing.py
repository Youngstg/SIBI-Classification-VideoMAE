"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/random_erasing.py
pulished under an Apache License 2.0.
"""
import math
import random
import torch


def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device="cuda"):
    """Generate random pixel values for erasing patch.
    
    Args:
        per_pixel: If True, each pixel has independent random value
        rand_color: If True, per-channel random color
        patch_size: Size of the patch to fill
    """
    if per_pixel:
        # Per-pixel random (independent normal distribution)
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        # Per-channel random (same value across spatial dimensions)
        return torch.empty(
            (patch_size[0], 1, 1), dtype=dtype, device=device
        ).normal_()
    else:
        # Constant (zero) value
        return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """Randomly erase rectangular regions in images/videos (augmentation).
    
    Reference: 'Random Erasing Data Augmentation' by Zhong et al.
    https://arxiv.org/pdf/1708.04896.pdf
    
    Applies to normalized tensors (after dataset normalization).
    
    Args:
        probability: Chance of applying random erasing
        min_area: Minimum erased area (fraction of image)
        max_area: Maximum erased area (fraction of image)
        min_aspect: Minimum aspect ratio of erased region
        mode: Fill mode - 'const' (zeros), 'rand' (per-channel), 'pixel' (per-pixel)
        max_count: Maximum number of erasing regions per image
    """

    def __init__(
        self,
        probability=0.5,
        min_area=0.02,
        max_area=1 / 3,
        min_aspect=0.3,
        max_aspect=None,
        mode="const",
        min_count=1,
        max_count=None,
        num_splits=0,
        device="cuda",
        cube=True,
    ):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        self.cube = cube
        if mode == "rand":
            self.rand_color = True  # per block random normal
        elif mode == "pixel":
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == "const"
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        for _ in range(count):
            for _ in range(10):
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top : top + h, left : left + w] = _get_pixels(
                        self.per_pixel,
                        self.rand_color,
                        (chan, h, w),
                        dtype=dtype,
                        device=self.device,
                    )
                    break

    def _erase_cube(
        self,
        img,
        batch_start,
        batch_size,
        chan,
        img_h,
        img_w,
        dtype,
    ):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = (
            self.min_count
            if self.min_count == self.max_count
            else random.randint(self.min_count, self.max_count)
        )
        for _ in range(count):
            for _ in range(100):
                target_area = (
                    random.uniform(self.min_area, self.max_area) * area / count
                )
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    for i in range(batch_start, batch_size):
                        img_instance = img[i]
                        img_instance[
                            :, top : top + h, left : left + w
                        ] = _get_pixels(
                            self.per_pixel,
                            self.rand_color,
                            (chan, h, w),
                            dtype=dtype,
                            device=self.device,
                        )
                    break

    def __call__(self, input):
        if len(input.size()) == 3:
            self._erase(input, *input.size(), input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.size()
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = (
                batch_size // self.num_splits if self.num_splits > 1 else 0
            )
            if self.cube:
                self._erase_cube(
                    input,
                    batch_start,
                    batch_size,
                    chan,
                    img_h,
                    img_w,
                    input.dtype,
                )
            else:
                for i in range(batch_start, batch_size):
                    self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input
