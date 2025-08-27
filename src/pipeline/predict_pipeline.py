import os
import sys
import pandas as pd
import numpy as np
from utils.logger import get_logger
from utils.exception import CustomException
from utils.utils import load_object

logger = get_logger(__name__)

class PredictionPipeline:
    def __init__(self, model_path=None, preprocessor_path=None):
        """
        Initialize prediction pipeline
        
        Args:
            model_path: Path to trained model file
            preprocessor_path: Path to preprocessor object file
        """
        try:
            # If paths are not provided, use the latest model and preprocessor
            if model_path is None or preprocessor_path is None:
                artifacts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts")
                os.makedirs(artifacts_dir, exist_ok=True)
                
                # Get the latest timestamp directory
                timestamp_dirs = [d for d in os.listdir(artifacts_dir) if os.path.isdir(os.path.join(artifacts_dir, d))]
                if not timestamp_dirs:
                    raise Exception("No trained models found in artifacts directory")
                
                latest_dir = max(timestamp_dirs)
                
                if model_path is None:
                    model_path = os.path.join(artifacts_dir, latest_dir, "model_trainer", "model.pkl")
                
                if preprocessor_path is None:
                    preprocessor_path = os.path.join(artifacts_dir, latest_dir, "data_transformation", "preprocessed.pkl")
            
            logger.info(f"Loading model from: {model_path}")
            logger.info(f"Loading preprocessor from: {preprocessor_path}")
            
            self.model = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)
            
        except Exception as e:
            logger.error(f"Error initializing prediction pipeline: {e}")
            raise CustomException(e)
    
    def predict(self, features):
        """
        Make predictions on input features
        
        Args:
            features: Input features as DataFrame or array
            
        Returns:
            Predictions array
        """
        try:
            logger.info("Starting prediction")
            
            # Preprocess features
            if isinstance(features, pd.DataFrame):
                # Drop any unnamed/index-like columns that weren't present at fit time
                cleaned_cols = []
                for c in features.columns:
                    cname = str(c)
                    if cname.strip() == "" or cname.lower().startswith("unnamed"):
                        continue
                    cleaned_cols.append(c)
                if len(cleaned_cols) != len(features.columns):
                    features = features[cleaned_cols]
                    logger.info("Dropped unnamed/index columns before transform")
                # Remove Wafer column if present
                if "Wafer" in features.columns:
                    wafer_ids = features["Wafer"].values
                    features = features.drop(columns=["Wafer"])
                
                # Remove target column if present
                if "Good/Bad" in features.columns:
                    features = features.drop(columns=["Good/Bad"])
            
            # Transform features
            transformed_features = self.preprocessor.transform(features)
            
            # Make predictions
            predictions = self.model.predict(transformed_features)
            
            logger.info("Prediction completed successfully")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise CustomException(e)


class CustomData:
    """
    Class to convert user input data to DataFrame for prediction
    """
    def __init__(self, **kwargs):
        """
        Initialize with sensor values
        """
        self.sensor_data = kwargs
    
    def get_data_as_dataframe(self):
        """
        Convert input data to DataFrame
        
        Returns:
            Pandas DataFrame
        """
        try:
            return pd.DataFrame(self.sensor_data, index=[0])
        except Exception as e:
            logger.error(f"Error converting data to DataFrame: {e}")
            raise CustomException(e)