import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Use relative imports when deployed
try:
    from src.utils.logger import get_logger
    from src.utils.exception import CustomException
    from src.utils.utils import save_object
    from src.config.config import DATA_TRANSFORMATION_CONFIG
except ImportError:
    # Fallback to direct imports if src module is not found
    from utils.logger import get_logger
    from utils.exception import CustomException
    from utils.utils import save_object
    from config.config import DATA_TRANSFORMATION_CONFIG

logger = get_logger(__name__)

class DataTransformation:
    def __init__(self):
        """
        Initialize data transformation with configuration
        """
        self.transformation_config = DATA_TRANSFORMATION_CONFIG
    
    def get_data_transformer_object(self):
        """
        Create preprocessing pipeline for numerical features
        
        Returns:
            Sklearn Pipeline object
        """
        try:
            # Create preprocessing pipeline for numerical features
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            logger.info("Numerical preprocessing pipeline created")
            return numerical_pipeline
            
        except Exception as e:
            logger.error(f"Error in creating data transformer: {e}")
            raise CustomException(e)
    
    def initiate_data_transformation(self, train_file_path, test_file_path):
        """
        Initiate the data transformation process
        
        Args:
            train_file_path: Path to training data
            test_file_path: Path to test data
            
        Returns:
            Tuple of transformed train and test data arrays and preprocessor object path
        """
        try:
            # Create directories
            os.makedirs(self.transformation_config["transformed_train_dir"], exist_ok=True)
            os.makedirs(self.transformation_config["transformed_test_dir"], exist_ok=True)
            
            # Read data
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            logger.info(f"Train dataset shape: {train_df.shape}")
            logger.info(f"Test dataset shape: {test_df.shape}")
            
            # Separate target feature
            target_column = "Good/Bad"
            
            input_feature_train_df = train_df.drop(columns=[target_column, "Wafer"], axis=1)
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=[target_column, "Wafer"], axis=1)
            target_feature_test_df = test_df[target_column]
            
            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()
            
            # Transform features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            # Combine features and target
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Save transformed data
            transformed_train_file = os.path.join(self.transformation_config["transformed_train_dir"], "transformed_train.npz")
            transformed_test_file = os.path.join(self.transformation_config["transformed_test_dir"], "transformed_test.npz")
            
            np.savez_compressed(transformed_train_file, data=train_arr)
            np.savez_compressed(transformed_test_file, data=test_arr)
            
            logger.info(f"Transformed train data saved at: {transformed_train_file}")
            logger.info(f"Transformed test data saved at: {transformed_test_file}")
            
            # Save preprocessing object
            preprocessor_file_path = self.transformation_config["preprocessed_object_file_path"]
            save_object(preprocessor_file_path, preprocessing_obj)
            
            logger.info(f"Preprocessor object saved at: {preprocessor_file_path}")
            
            return (
                train_arr,
                test_arr,
                preprocessor_file_path
            )
            
        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise CustomException(e)