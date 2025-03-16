# Corn Mycotoxin Level Prediction

This project implements a machine learning pipeline to predict DON (Deoxynivalenol) concentration in corn samples using hyperspectral imaging data.

## Project Structure
```
├── data_preprocessing.py   # Data preprocessing and feature engineering
├── model.py               # Neural network model definition and training
├── utils.py              # Utility functions
├── evaluation.py         # Model evaluation metrics and visualization
├── api.py               # FastAPI server for model deployment
└── requirements.txt     # Project dependencies
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:
```bash
python data_preprocessing.py
```

2. Train Model:
```bash
python model.py
```

3. Run API Server:
```bash
uvicorn api:app --reload
```

## API Endpoints

- `POST /predict`: Upload a CSV file containing spectral data to get DON concentration predictions
  - Input: CSV file with wavelength features (448 columns)
  - Output: Predicted DON concentration

## Model Details

- Architecture: Neural Network Regressor
- Input: 448 spectral wavelength features
- Output: DON concentration prediction
- Evaluation Metrics: MAE, RMSE, R² Score

## Data Format

The input CSV should contain spectral reflectance values across 448 wavelength bands (columns 0-447). Each row represents one corn sample. 