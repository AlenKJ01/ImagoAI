import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def load_data(self, file_path, nrows=50):
        """Load data from CSV file."""
        try:
            logger.info(f"Loading data from {file_path}")
            df = pd.read_csv(file_path, nrows=nrows)
            logger.info(f"Loaded {len(df)} samples")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self, df, is_training=True):
        """Preprocess the data for training or prediction."""
        try:
            # Remove hsi_id if present
            if 'hsi_id' in df.columns:
                df = df.drop('hsi_id', axis=1)
            
            # For training data
            if is_training:
                # Separate features and target
                X = df.drop('vomitoxin_ppb', axis=1) if 'vomitoxin_ppb' in df.columns else df
                y = df['vomitoxin_ppb'] if 'vomitoxin_ppb' in df.columns else None
                
                # Store feature columns for future use
                self.feature_columns = X.columns.tolist()
                
                # Handle missing values
                X = self._handle_missing_values(X)
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
                
                # Create visualizations
                self._create_visualizations(df, X_scaled, y)
                
                return X_scaled, y
                
            # For prediction data
            else:
                # Load feature columns if not loaded
                if self.feature_columns is None:
                    self.load_feature_columns()
                    
                # Reorder columns to match training data
                if len(df.columns) != 448:
                    raise ValueError(f"Expected 448 features, got {len(df.columns)}")
                    
                # Assign column names if they don't exist
                if df.columns[0] != '0':
                    df.columns = [str(i) for i in range(448)]
                
                X = df[self.feature_columns] if self.feature_columns else df
                
                # Handle missing values
                X = self._handle_missing_values(X)
                
                # Scale features
                X_scaled = self.scaler.transform(X)
                X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
                
                return X_scaled
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise
                
    def _handle_missing_values(self, X):
        """Handle missing values in the data."""
        # Fill missing values with median of the column
        return X.fillna(X.median())
        
    def _create_visualizations(self, df, X_scaled, y):
        """Create visualizations for data analysis."""
        try:
            os.makedirs('plots', exist_ok=True)
            
            # Plot target distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(y, bins=30)
            plt.title('Distribution of DON Concentration')
            plt.xlabel('DON Concentration (ppb)')
            plt.ylabel('Count')
            plt.savefig('plots/target_distribution.png')
            plt.close()
            
            # Plot average spectral signature
            plt.figure(figsize=(12, 6))
            mean_spectrum = X_scaled.mean()
            std_spectrum = X_scaled.std()
            wavelengths = range(len(mean_spectrum))
            plt.plot(wavelengths, mean_spectrum)
            plt.fill_between(wavelengths, 
                           mean_spectrum - std_spectrum,
                           mean_spectrum + std_spectrum,
                           alpha=0.2)
            plt.title('Average Spectral Signature')
            plt.xlabel('Wavelength Index')
            plt.ylabel('Normalized Reflectance')
            plt.savefig('plots/average_spectrum.png')
            plt.close()
            
            logger.info("Visualizations created successfully")
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            logger.warning("Continuing without visualizations")
        
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    def save_scaler(self, path='models/scaler.joblib'):
        """Save the fitted scaler."""
        try:
            import joblib
            import os
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.scaler, path)
            logger.info(f"Scaler saved to {path}")
            
            # Save feature columns
            feature_columns_path = os.path.join(os.path.dirname(path), 'feature_columns.joblib')
            joblib.dump(self.feature_columns, feature_columns_path)
            logger.info(f"Feature columns saved to {feature_columns_path}")
        except Exception as e:
            logger.error(f"Error saving scaler or feature columns: {str(e)}")
            raise
        
    def load_scaler(self, path='models/scaler.joblib'):
        """Load the fitted scaler."""
        try:
            import joblib
            self.scaler = joblib.load(path)
            logger.info(f"Scaler loaded from {path}")
            
            # Load feature columns
            self.load_feature_columns()
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            raise
            
    def load_feature_columns(self, path='models/feature_columns.joblib'):
        """Load the feature columns."""
        try:
            import joblib
            self.feature_columns = joblib.load(path)
            logger.info(f"Feature columns loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading feature columns: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data("MLE-Assignment.csv", nrows=50)
    X_scaled, y = preprocessor.preprocess_data(df, is_training=True)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_scaled, y)
    preprocessor.save_scaler()
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Testing set shape: {X_test.shape}") 