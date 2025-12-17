from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import uvicorn
import cv2
import numpy as np
from src.pipeline.segment import SegmentationModule
from src.pipeline.layer_detection import LayerDetectionModule

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
    # Receive raw pixels from QuPath
    data = await request.body()
    # Assuming pixels are sent as a byte array (PNG/JPG)
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Hardcoded task for this example; you can pass this in headers or params
    task = request.query_params.get("task", "nuclei")
    
    features = []
    if task == 'tissue':
            mask = seg_module.segment_tissue(img_rgb)
        # ... (use the mask_to_features logic from before)
    else:
        data_dict = seg_module.segment_nuclei(img_rgb)
        # ... (use the stardist_to_features logic from before)
        
    return {"type": "FeatureCollection", "features": features}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)