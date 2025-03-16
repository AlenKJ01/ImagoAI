import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime

def test_prediction():
    # File to send
    file_path = 'MLE-Assignment.csv'
    
    try:
        # Read and preprocess the data
        df = pd.read_csv(file_path, header=None)
        
        # Select only the required 448 features (excluding the first two columns)
        df = df.iloc[:, 2:450]
        
        # Save preprocessed data
        temp_file = 'temp_preprocessed.csv'
        df.to_csv(temp_file, index=False, header=False)
        
        # Prepare the file for upload
        files = {'file': open(temp_file, 'rb')}
        
        # Make the POST request
        response = requests.post('http://127.0.0.1:8000/predict', files=files)
        
        # Print the response
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            predictions = np.array(result['predictions']).flatten()
            print("\nPredictions Summary:")
            print(f"Number of predictions: {len(predictions)}")
            print(f"Mean DON concentration: {np.mean(predictions):.2f}")
            print(f"Min DON concentration: {np.min(predictions):.2f}")
            print(f"Max DON concentration: {np.max(predictions):.2f}")
            print(f"\nFirst 5 predictions: {predictions[:5]}")
        else:
            print("\nError Response:")
            print(json.dumps(response.json(), indent=2))
        
        # Clean up
        import os
        os.remove(temp_file)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    
if __name__ == "__main__":
    test_prediction() 