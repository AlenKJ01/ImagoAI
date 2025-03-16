from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io
import logging
import json
from pathlib import Path
from data_preprocessing import DataPreprocessor
from model import SpectralModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Corn DON Concentration Predictor")

# Initialize preprocessor and model
preprocessor = DataPreprocessor()
model = SpectralModel()

# Create predictions directory if it doesn't exist
PREDICTIONS_DIR = Path("predictions")
PREDICTIONS_DIR.mkdir(exist_ok=True)

# Load the trained model and scaler
try:
    model.load_model()
    preprocessor.load_scaler()
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to predict DON concentration from spectral data
    
    Expects a CSV file with 448 columns of spectral data (no header)
    Returns predicted DON concentration and saves results to a JSON file
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Preprocess the data
        X_scaled = preprocessor.preprocess_data(df, is_training=False)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Create results dictionary
        results = {
            "status": "success",
            "input_file": file.filename,
            "predictions": predictions.tolist(),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Save predictions to JSON file
        output_file = PREDICTIONS_DIR / f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Predictions saved to {output_file}")
        
        return JSONResponse({
            **results,
            "output_file": str(output_file)
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=400)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 