import hydra
import joblib
import tifffile
import numpy as np
import sys
from pathlib import Path
from skimage import color, filters, feature
from skimage.util import img_as_ubyte
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
# Add project root to path
script_path = Path(__file__).resolve()
src_dir = script_path.parent
project_root = src_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.helpers import get_image_and_mask_paths



class RandomForestWrapper():
    def __init__(self, cfg: DictConfig):
        model = hydra.utils.instantiate(cfg.model.params)
        self.model = model
        self.cfg = cfg

        data_dir = cfg.data.data_dir
        mask_dir = cfg.data.mask_dir

        self.train_image_paths, self.train_mask_paths, \
        self.test_image_paths, self.test_mask_paths = \
            get_image_and_mask_paths(data_dir, mask_dir)
        

        self.num_classes = cfg.model.get('num_classes', 3)
        self.binary_mode = self.num_classes == 2

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
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

    def _convert_to_binary(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert 3-class mask to binary (background vs tissue).
        
        Args:
            mask: Mask with values {0, 1, 2}
        
        Returns:
            Binary mask with values {0, 1}
        """
        return (mask > 0).astype(mask.dtype)

    def prepare_data(self, subsample_rate: float = 1.0):
        """
        Prepare training and test data by extracting features and flattening.
        
        Args:
            subsample_rate (float): Fraction of pixels to use (1.0 = all pixels). Defaults to 1.0.
        """
        print("\nPreparing training data...")
        X_train_list = []
        y_train_list = []
        
        for img_path, mask_path in tqdm(
            zip(self.train_image_paths, self.train_mask_paths),
            total=len(self.train_image_paths),
            desc="Processing training images"
        ):
            image = tifffile.imread(img_path)
            mask = tifffile.imread(mask_path)
            
            if self.binary_mode:
                mask = self._convert_to_binary(mask)
            
            features = self._extract_features(image)
            
            features_flat = features.reshape(-1, features.shape[-1])
            mask_flat = mask.flatten()
            
            if subsample_rate < 1.0:
                n_pixels = len(mask_flat)
                n_samples = int(n_pixels * subsample_rate)
                indices = np.random.choice(n_pixels, n_samples, replace=False)
                features_flat = features_flat[indices]
                mask_flat = mask_flat[indices]
            
            X_train_list.append(features_flat)
            y_train_list.append(mask_flat)
        
        self.X_train = np.vstack(X_train_list)
        self.y_train = np.concatenate(y_train_list)
        
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Training labels shape: {self.y_train.shape}")
        
        print("\nPreparing test data...")
        X_test_list = []
        y_test_list = []
        
        for img_path, mask_path in tqdm(
            zip(self.test_image_paths, self.test_mask_paths),
            total=len(self.test_image_paths),
            desc="Processing test images"
        ):
            image = tifffile.imread(img_path)
            mask = tifffile.imread(mask_path)
            
            if self.binary_mode:
                mask = self._convert_to_binary(mask)
            
            features = self._extract_features(image)
            features_flat = features.reshape(-1, features.shape[-1])
            mask_flat = mask.flatten()
            
            X_test_list.append(features_flat)
            y_test_list.append(mask_flat)
        
        self.X_test = np.vstack(X_test_list)
        self.y_test = np.concatenate(y_test_list)
        
        print(f"Test data shape: {self.X_test.shape}")
        print(f"Test labels shape: {self.y_test.shape}")
    

    def fit(self):
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        self.model.fit(self.X_train, self.y_train)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict segmentation mask for a single image.
        
        Args:
            image (np.ndarray): Input RGB image of shape (H, W, 3)
            
        Returns:
            Predicted mask of shape (H, W) with class labels
        """
        features = self._extract_features(image)
        
        height, width = image.shape[:2]
        
        features_flat = features.reshape(-1, features.shape[-1])
        predictions_flat = self.model.predict(features_flat)
        predictions = predictions_flat.reshape(height, width)
        
        return predictions

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Enable calling the model directly: model(image).
        
        Args:
            image (np.ndarray): Input RGB image of shape (H, W, 3)
            
        Returns:
            Predicted mask of shape (H, W) with class labels
        """
        return self.predict(image)

    def evaluate(self, mode: str = "test") -> tuple[float, float]:
        if mode == "train":
            if self.X_train is None:
                raise ValueError("Train data not prepared. Call prepare_data() first.")
            y_pred = self.predict(self.X_train)
            y_gt = self.y_train

        else:
            if self.X_test is None:
                raise ValueError("Test data not prepared. Call prepare_data() first.")
            y_pred = self.predict(self.X_test)
            y_gt = self.y_test
        
        accuracy = accuracy_score(y_gt, y_pred)
        f1 = f1_score(y_gt, y_pred, average=self.cfg.eval.f1_avg)
        
        if self.binary_mode:
            f1_binary = f1_score(y_gt, y_pred, average="binary")
        else:
            f1_binary = None
        
        cm = confusion_matrix(y_gt, y_pred)
        
        results = {
            "accuracy": accuracy,
            "f1": f1,
            "f1_binary": f1_binary,
            "confusion_matrix": cm,
        }
        return results

    def save(self, path: Path, name: str = None) -> Path:
        if not name:
            name = self.cfg.model.name
        
        path.mkdir(parents=True, exist_ok=True)

        model_path = path / f"{name}.joblib"

        joblib.dump(self.model, model_path)
        print(f"Model saved to: {model_path}")
        return model_path

    def load(self, path: Path):
        """Load a trained model from disk."""
        self.model = joblib.load(path)
        print(f"Model loaded from: {path}")