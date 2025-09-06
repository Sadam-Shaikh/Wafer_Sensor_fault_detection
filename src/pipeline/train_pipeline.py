import os
import sys
import pandas as pd

# Use relative imports when deployed
try:
    from src.components.data_ingestion import DataIngestion
    from src.components.data_validation import DataValidation
    from src.components.data_transformation import DataTransformation
    from src.components.model_trainer import ModelTrainer
    from src.utils.logger import get_logger
    from src.utils.exception import CustomException
    from src.utils.mongodb_utils import MongoDBClient
except ImportError:
    # Fallback to direct imports if src module is not found
    from components.data_ingestion import DataIngestion
    from components.data_validation import DataValidation
    from components.data_transformation import DataTransformation
    from components.model_trainer import ModelTrainer
    from utils.logger import get_logger
    from utils.exception import CustomException
    from utils.mongodb_utils import MongoDBClient

logger = get_logger(__name__)

class TrainPipeline:
    def __init__(self):
        """
        Initialize training pipeline
        """
        self.data_ingestion = DataIngestion()
        self.data_validation = DataValidation()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
    
    def start_training(self, data_file_path=None):
        """
        Start the training pipeline
        
        Args:
            data_file_path: Path to input data file (optional)
            
        Returns:
            Path to trained model
        """
        try:
            logger.info("Starting training pipeline")
            
            # Data Ingestion
            logger.info("Data Ingestion started")
            train_file_path, test_file_path = self.data_ingestion.initiate_data_ingestion(data_file_path)
            logger.info("Data Ingestion completed")
            
            # Data Validation
            logger.info("Data Validation started")
            validation_status = self.data_validation.initiate_data_validation(train_file_path, test_file_path)
            logger.info("Data Validation completed")
            
            if not validation_status:
                raise Exception("Data validation failed. Please check the data.")
            
            # Data Transformation
            logger.info("Data Transformation started")
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(
                train_file_path, test_file_path
            )
            logger.info("Data Transformation completed")
            
            # Model Training
            logger.info("Model Training started")
            model_path = self.model_trainer.initiate_model_training(train_arr, test_arr)
            logger.info("Model Training completed")
            
            # Save model metadata to MongoDB
            try:
                # Connect to MongoDB
                mongo_client = MongoDBClient()
                
                # Create model metadata
                model_metadata = {
                    "model_path": model_path,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "preprocessor_path": preprocessor_path,
                    "status": "trained"
                }
                
                # Save model metadata
                if mongo_client.test_connection():
                    mongo_client.save_dataframe(
                        pd.DataFrame([model_metadata]), 
                        "models", 
                        metadata={"type": "model_metadata"}
                    )
                
                logger.info("Model metadata saved to MongoDB")
            except Exception as e:
                logger.warning(f"Failed to save model metadata to MongoDB: {e}")
                logger.info("Continuing with local model storage only")
            
            logger.info(f"Training pipeline completed successfully. Model saved at: {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise CustomException(e)


if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    model_path = train_pipeline.start_training()
    print(f"Model saved at: {model_path}")