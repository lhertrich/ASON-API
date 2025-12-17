import numpy as np
import torch

from skimage.exposure import rescale_intensity
from skimage.transform import resize
from stardist.models import StarDist2D
from src.data.preprocessing import inference_processing
from src.models.model_loader import ModelLoader
from src.utils.reinhard_normalizer import ReinhardNormalizer


class SegmentationModule:
    def __init__(
        self, tissue_model_name: str = "unet_2c", tissue_model_config: str = "unet_2"
    ) -> None:
        model_loader = ModelLoader()

        self.normalizer = ReinhardNormalizer()
        self.tissue_model = model_loader.load_cnn_model(
            tissue_model_config, tissue_model_name
        )
        self.nuclei_model = StarDist2D.from_pretrained("2D_versatile_he")
        self.device = self._get_device()

    def segment_tissue(
        self, image: np.ndarray, org_res: tuple[int, int] = (1920, 2560)
    ) -> np.ndarray:
        img = inference_processing(image, self.device)

        with torch.no_grad():
            pred_logits = self.tissue_model(img)
            pred_mask = torch.argmax(pred_logits, dim=1).squeeze()
            pred_mask = pred_mask.cpu().numpy()
            pred_mask = resize(pred_mask, org_res, anti_aliasing=True)

        return pred_mask

    def segment_nuclei(
        self,
        image: np.ndarray,
        prob_thresh: float = 0.25,
        nms_thresh: float = 0.01,
        cleaned: bool = True,
        area_th: float = 0.5,
    ) -> np.ndarray:
        image_normed = rescale_intensity(image, out_range=(0, 1))
        _, data_dict = self.nuclei_model.predict_instances(
            image_normed,
            axes="YXC",
            prob_thresh=0.25,
            nms_thresh=0.01,
            return_labels=True,
        )
        if cleaned:
            mask = self.segment_tissue(image)
            data_dict = self._filter_data_dict(mask, data_dict, area_th)

        return data_dict

    def _filter_data_dict(
        self, mask: np.ndarray, data_dict: dict[str, any], area_th: float = 0.5
    ) -> list:
        points = data_dict["points"]
        median_area = self._calculate_median_area(data_dict["coord"])
        filtered_points = []
        filtered_coords = []
        filtered_probs = []

        binary_mask = (mask > 0).astype(int)
        for i, (point, coord) in enumerate(zip(points, data_dict["coord"])):
            x, y = int(point[0]), int(point[1])
            area = self._poly_area(np.array(coord[0]), np.array(coord[1]))
            if binary_mask[x, y] == 1 and area > area_th * median_area:
                filtered_points.append([point[0], point[1]])
                filtered_coords.append(coord)
                filtered_probs.append(data_dict["prob"][i])

        filtered_data_dict = dict(data_dict)
        filtered_data_dict["points"] = np.array(filtered_points)
        filtered_data_dict["coord"] = np.array(filtered_coords)
        filtered_data_dict["prob"] = np.array(filtered_probs)

        return filtered_data_dict

    def _poly_area(self, x: int, y: int):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _calculate_median_area(self, coordinates: np.ndarray) -> float:
        areas = []
        for coord in coordinates:
            area = self._poly_area(np.array(coord[0]), np.array(coord[1]))
            areas.append(area)

        median_area = np.median(np.array(areas))
        return median_area

    def _get_device(self) -> str:
        device = "cpu"
        if torch.mps.is_available():
            device = "mps"
        if torch.cuda.is_available():
            device = "cuda"
        return device
