import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import random
import sys
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    disk,
)
from skimage.transform import resize
from pathlib import Path

script_path = Path(__file__).resolve()
src_path = script_path.parent.parent
project_root = src_path.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def clean_mask(mask, num_classes=2, min_size=200, area_threshold=10, radius=2):
    """
    Clean a multi-class segmentation mask.

    Args:
        mask: numpy array with integer class labels (0, 1, 2, ...)
        num_classes: number of classes in the mask
        min_size: minimum size of objects to keep
        area_threshold: maximum size of holes to fill
        radius: radius of morphological closing disk

    Returns:
        Cleaned mask with same shape and dtype as input
    """
    cleaned_mask = np.zeros_like(mask, dtype=mask.dtype)

    for cls in range(num_classes):
        binary_mask = mask == cls

        if not binary_mask.any():
            continue

        cleaned_class = remove_small_objects(binary_mask, min_size=min_size)
        cleaned_class = remove_small_holes(cleaned_class, area_threshold=area_threshold)

        rad = disk(radius=radius)
        cleaned_class = binary_closing(cleaned_class, rad)

        cleaned_mask[cleaned_class] = cls

    return cleaned_mask


def cut_out_image(image, mask):
    resized_mask = resize(
        mask, (image.shape[0], image.shape[1]), order=0, preserve_range=True
    )
    cut_image = image.copy()
    resized_mask = resized_mask.astype(bool)
    cut_image[~resized_mask] = 0
    return cut_image


def compare_two_images(image_1, image_2, title_1=None, title_2=None, size=(12, 6)):
    plt.figure(figsize=size)

    plt.subplot(1, 2, 1)
    plt.axis(False)
    plt.title(title_1)
    plt.imshow(image_1)

    plt.subplot(1, 2, 2)
    plt.axis(False)
    plt.title(title_2)
    plt.imshow(image_2)

    plt.show()


def get_image_and_mask_paths(data_dir, mask_dir):
    image_paths_train = sorted(
        [
            data_dir + "/" + path
            for path in os.listdir(data_dir)
            if not path.startswith(".")
        ]
    )
    mask_paths_train = sorted(
        [
            mask_dir + "/" + path
            for path in os.listdir(mask_dir)
            if not path.startswith(".")
        ]
    )
    image_paths_test = sorted(
        [
            data_dir + "_test/" + path
            for path in os.listdir(data_dir + "_test")
            if not path.startswith(".")
        ]
    )
    mask_paths_test = sorted(
        [
            mask_dir + "_test/" + path
            for path in os.listdir(mask_dir + "_test")
            if not path.startswith(".")
        ]
    )
    return image_paths_train, mask_paths_train, image_paths_test, mask_paths_test


def seed_everything(seed: int = 42):
    """
    Seed all random number generators for reproducibility.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    print(f"All random seeds set to: {seed}")