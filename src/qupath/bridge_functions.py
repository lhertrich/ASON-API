import cv2
import numpy as np

def mask_to_polygons(mask: np.ndarray, min_area=5000):
    """
    Convert binary mask to list of polygons (pixel coordinates)
    """
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        cnt = cnt.squeeze()
        if cnt.ndim != 2:
            continue

        # GeoJSON requires closed rings
        ring = cnt.tolist()
        ring.append(ring[0])

        polygons.append(ring)

    return polygons
