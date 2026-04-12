import io
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

# We'll use the existing preprocessing and inference logic for simplicity
# In a real production setup, one might prefer ONNX directly.
from src.inference.predict import run_sliding_window_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CloudShadow-UNet API",
    description="Geospatial semantic segmentation for Clouds and Shadows.",
    version="1.0.0"
)

# Global model cache attached to the app state
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_weights.h5")
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", 256))
OVERLAP = float(os.environ.get("OVERLAP", 0.25))

@app.on_event("startup")
def load_model():
    if not Path(MODEL_PATH).exists():
        logger.warning(f"Model path {MODEL_PATH} not found. Ensure it is mapped or exists before calling /predict")
        app.state.model = None
    else:
        logger.info(f"Loading Keras model from {MODEL_PATH}...")
        # Compile=False is fine for inference
        app.state.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Model loaded successfully.")

@app.get("/health")
def health_check():
    """Returns the operational status of the service and model."""
    status = "healthy" if app.state.model is not None else "model_missing"
    return {"status": status, "model_path": MODEL_PATH}

@app.post("/predict")
async def predict_geotiff(file: UploadFile = File(...)):
    """Accepts a 4-band GeoTIFF, runs inference, and returns a Masked GeoTIFF."""
    if not file.filename.endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .tif or .tiff GeoTIFF file.")
    
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Cannot process request.")
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_in:
        content = await file.read()
        tmp_in.write(content)
        tmp_in_path = tmp_in.name
    
    # Define an output temporary file
    tmp_out_path = tmp_in_path.replace(".tif", "_mask.tif")
    
    try:
        # We reuse the `run_sliding_window_inference` logic from the existing predict module.
        # However, because predict.py is script-focused, it's better to adapt its core or assume it's refactored.
        # Assuming run_sliding_window_inference is exported nicely:
        
        from src.inference.predict import sliding_window_predict
        from src.preprocessing.preprocess import read_multiband_geotiff
        
        # 1. Read the georeferenced array
        image_array, profile = read_multiband_geotiff(Path(tmp_in_path))
        
        # 2. Run inference (sliding_window_predict expects a list of models now)
        mask = sliding_window_predict(
            models=[app.state.model],
            image=image_array,
            patch_size=PATCH_SIZE,
            overlap=OVERLAP
        )
        
        # 3. Write output to temporary file
        out_profile = profile.copy()
        out_profile.update({
            "count": 1,
            "dtype": "uint8"
        })
        
        with rasterio.open(tmp_out_path, "w", **out_profile) as dest:
            dest.write(mask.astype(np.uint8), 1)
            
        # Optional: return statistics alongside the file? 
        # Standard approach for returning files in FastAPI:
        return FileResponse(tmp_out_path, media_type="image/tiff", filename=f"mask_{file.filename}")

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    # Note: tmp files will be left on disk until cleared by OS, in production consider BackgroundTasks to delete.
