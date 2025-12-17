import numpy as np
import tifffile

from skimage import color, filters, feature
from skimage.util import img_as_ubyte
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_closing,
    disk,
)
from joblib import load


class MaskCreator:
    """Class for creating and refining masks from H&E images using a pre-trained random forest model.

    This class provides methods for feature extraction, mask prediction, mask cleaning,
    and saving the resulting mask as a file.

    Attributes:
        model: The loaded classification model used for pixel-wise prediction.
    """

    def __init__(self, model_path: str) -> None:
        """Initializes MaskCreator by loading a pre-trained model.

        Args:
            model_path (str): File path to the pre-trained model.

        Raises:
            ValueError: If the model cannot be loaded from the given path.
        """
        try:
            self.model = load(model_path)
        except Exception as e:
            raise ValueError(f"Failed to load mask from {model_path}: {e}")

    def __extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extracts features (color, texture, edge) per pixel from the input image.

        Args:
            image (np.ndarray): Input image as a NumPy array.

        Returns:
            np.ndarray: Stacked feature array of shape (H, W, num_features).
        """
        img_lab = color.rgb2lab(image)
        gray = color.rgb2gray(image)
        gray_uint8 = img_as_ubyte(gray)

        features = []
        # Raw color channels
        for i in range(3):
            features.append(image[..., i])
        for i in range(3):
            features.append(img_lab[..., i])
        # Smoothed color (Gaussian)
        for sigma in [1, 3]:
            features.append(filters.gaussian(gray, sigma))
        # Edges
        features.append(filters.sobel(gray))
        # Local Binary Pattern (texture)
        lbp = feature.local_binary_pattern(gray_uint8, P=8, R=1, method="uniform")
        features.append(lbp)
        feat_stack = np.stack(features, axis=-1)
        return feat_stack

    def __clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Cleans the predicted mask by removing small objects, filling holes, and smoothing edges.

        Args:
            mask (np.ndarray): Binary or multiclass mask as a NumPy array.

        Returns:
            np.ndarray: Refined binary mask after morphological operations.
        """
        mask = mask.astype(bool)
        mask = remove_small_objects(mask, min_size=10000)
        mask = remove_small_holes(mask, area_threshold=1000)
        rad = disk(radius=2)
        smoothed_mask = binary_closing(mask, rad)
        return smoothed_mask

    def __predict_mask(self, image: np.ndarray) -> np.ndarray:
        """Predicts a pixelwise mask (0, 1, 2) for an H&E image.

        Args:
            image (np.ndarray): Input H&E stained image as a NumPy array.

        Returns:
            np.ndarray: Predicted mask with labels for each pixel.
        """
        feat_stack = self.__extract_features(image)
        H, W, F = feat_stack.shape
        X = feat_stack.reshape(-1, F)
        y_pred = self.model.predict(X)
        pred_mask = y_pred.reshape(H, W)
        return pred_mask

    def create_mask(
        self, image: np.ndarray, save: bool = False, save_path: str = None
    ) -> np.ndarray:
        """Creates a mask for the input image, cleans it, and optionally saves it to a file.

        Args:
            image (np.ndarray): Input image as a NumPy array.
            save (bool, optional): Whether to save the mask to a file. Defaults to False.
            save_path (str, optional): File path to save the mask if 'save' is True. Defaults to None.

        Returns:
            np.ndarray: The cleaned predicted mask.

        Raises:
            ValueError: If 'save' is True but no 'save_path' is provided.
        """
        pred_mask = self.__predict_mask(image)
        pred_mask = self.__clean_mask(pred_mask)

        if save:
            if save_path:
                tifffile.imwrite(save_path, pred_mask)
            else:
                raise ValueError("Save path is required if save is True")

        return pred_mask
