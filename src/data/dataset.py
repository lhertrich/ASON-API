from torch.utils.data import Dataset
import tifffile
from typing import Optional, Callable, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PixelClassificationDataset(Dataset):
    """A dataset for pixel-wise classification from segmented images.

    Args:
        Dataset (torch.utils.data.Dataset): The parent class.
    """

    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        transform: Optional[Callable] = None,
        augmentations_per_image: int = 1,
        include_original: bool = True,
        binary_mode: bool = False,
    ):
        """
        Args:
            image_paths: List of paths to input images
            mask_paths: List of paths to segmentation masks
            transform: Albumentations transform pipeline
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augmentations_per_image = augmentations_per_image
        self.include_original = include_original
        self.binary_mode = binary_mode

        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose(
                [A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]), ToTensorV2()]
            )

        self.to_tensor = A.Compose(
            [A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]), ToTensorV2()]
        )

        assert len(image_paths) == len(mask_paths), "Mismatch between images and masks"

    def __len__(self):
        return len(self.image_paths) * self.augmentations_per_image

    def _convert_to_binary(self, mask):
        """
        Convert 3-class mask to binary mask.
        Class 0 (background) -> 0
        Class 1 and 2 (tissue types) -> 1

        Args:
            mask: Mask with values {0, 1, 2}

        Returns:
            Binary mask with values {0, 1}
        """
        binary_mask = (mask > 0).astype(mask.dtype)
        return binary_mask

    def __getitem__(self, idx):
        real_idx = idx // self.augmentations_per_image
        aug_idx = idx % self.augmentations_per_image

        image = tifffile.imread(self.image_paths[real_idx])
        mask = tifffile.imread(self.mask_paths[real_idx])

        # Convert to binary if specified
        if self.binary_mode:
            mask = self._convert_to_binary(mask)

        # First version is always original (no augmentation)
        if self.include_original and aug_idx == 0:
            transformed = self.to_tensor(image=image, mask=mask)
        else:
            # Apply random augmentation
            transformed = self.transform(image=image, mask=mask)

        return transformed["image"], transformed["mask"].long()
