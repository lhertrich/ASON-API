import numpy as np
import cv2
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

TARGET_SIZE = (512, 512)
TRANSFORM = A.Compose(
                [A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]), ToTensorV2()]
            )

def inference_processing(target_image: np.ndarray, device="cpu") -> torch.Tensor:
    """Provides the same preprocessing as the training pipeline for inference.

    Args:
        target_image (np.ndarray): The target image to preprocess
        device (str, optional): The device to use for the tensor. Defaults to "cpu".

    Returns:
        torch.Tensor: The preprocessed target image as a tensor
    """
    resized_image = cv2.resize(target_image, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)
    transformed = TRANSFORM(image=resized_image)
    tensor_image = transformed["image"]
    tensor_image = tensor_image.unsqueeze(0)
    tensor_image = tensor_image.to(device)
    return tensor_image