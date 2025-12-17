from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import uvicorn
import cv2
import numpy as np
from src.pipeline.segment import SegmentationModule
from src.pipeline.layer_detection import LayerDetectionModule
from src.qupath.bridge_functions import mask_to_polygons

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing application and loadingmodels...")
    app.state.segmentation = SegmentationModule()
    app.state.layer_detection = LayerDetectionModule()
    print("Application startup complete")
    
    yield
    
    print("Shutting down...")
    del app.state.segmentation
    del app.state.layer_detection

app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "online", "modules_loaded": ["segmentation", "layer_detection"]}


@app.post("/segment")
async def segment(request: Request):
    seg_module = request.app.state.segmentation

    data = await request.body()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    task = request.query_params.get("task", "tissue")
    mode = request.query_params.get("mode", "overlay")

    if task != "tissue":
        return {"error": "Only tissue supported in this endpoint"}

    mask = seg_module.segment_tissue(img_rgb)  # binary mask (H,W)

    polygons = mask_to_polygons(mask)

    features = []
    for poly in polygons:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [poly]
            },
            "properties": {
                "class": "Tissue",
                "mode": mode
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }

@app.post("/layer_detection")
async def layer_detection(request: Request):
    layer_detection_module = request.app.state.layer_detection
    data = await request.body()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    layers = layer_detection_module.detect_layers(img_rgb)
    return {"layers": layers}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)