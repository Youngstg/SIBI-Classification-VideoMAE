import numpy as np

class TubeMaskingGenerator:
    """Generate random masking pattern for VideoMAE pre-training.
    Applies uniform masking ratio across all frames (tube-wise masking).
    """
    
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width  # Patches per frame
        self.total_patches = self.frames * self.num_patches_per_frame  # Total patches
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)  # Masks per frame
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        """Generate binary mask (0=visible, 1=masked) for all frames."""
        # Create mask pattern: 0s for visible patches, 1s for masked patches
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        # Shuffle to randomize mask positions
        np.random.shuffle(mask_per_frame)
        # Apply same mask pattern across all frames (tube-wise)
        mask = np.tile(mask_per_frame, (self.frames, 1)).flatten()
        return mask 