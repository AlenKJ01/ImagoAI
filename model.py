import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import joblib
import os
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpectralModel:
    def __init__(self, input_dim=448):
        self.input_dim = input_dim
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the neural network model."""
        model = models.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)  # Output layer for regression
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=8):
        """Train the model."""
        logger.info("Starting model training...")
        
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Model checkpoint callback
        os.makedirs('models', exist_ok=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_loss',
            save_best_only=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        logger.info("Model training completed")
        return history
        
    def evaluate(self, X_test, y_test):
        """Evaluate the model and return metrics."""
        y_pred = self.model.predict(X_test, verbose=0)  # Added verbose=0 to reduce output
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info("Model Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
            
        return metrics, y_pred
        
    def plot_predictions(self, y_true, y_pred, save_path='plots'):
        """Plot actual vs predicted values."""
        os.makedirs(save_path, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual DON Concentration')
        plt.ylabel('Predicted DON Concentration')
        plt.title('Actual vs Predicted DON Concentration')
        plt.savefig(f'{save_path}/predictions.png')
        plt.close()
        
    def explain_predictions(self, X_train, X_test):
        """Generate SHAP values for model interpretability."""
        try:
            explainer = shap.KernelExplainer(self.model.predict, X_train[:10])  # Reduced sample size
            shap_values = explainer.shap_values(X_test[:10])
            
            plt.figure()
            shap.summary_plot(shap_values, X_test[:10], show=False)
            plt.savefig('plots/shap_summary.png')
            plt.close()
        except Exception as e:
            logger.error(f"Error generating SHAP values: {str(e)}")
            logger.warning("Continuing without SHAP analysis")
        
    def save_model(self, path='models/model.keras'):
        """Save the trained model."""
        self.model.save(path)
        logger.info(f"Model saved to {path}")
        
    def load_model(self, path='models/model.keras'):
        """Load a trained model."""
        self.model = models.load_model(path)
        logger.info(f"Model loaded from {path}")
        
    def predict(self, X):
        """Make predictions on new data."""
        return self.model.predict(X, verbose=0)

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data("MLE-Assignment.csv", nrows=50)  # Load only 50 samples
    X_scaled, y = preprocessor.preprocess_data(df, is_training=True)
    X_train, X_test, y_train, y_test = preprocessor.split_data(X_scaled, y)
    
    # Create and train model
    model = SpectralModel()
    history = model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    metrics, y_pred = model.evaluate(X_test, y_test)
    
    # Plot results
    model.plot_predictions(y_test, y_pred)
    
    # Generate explanations
    model.explain_predictions(X_train, X_test)
    
    # Save model
    model.save_model() 