import os
import json
import pandas as pd
from utils.logger import get_logger
from utils.exception import CustomException
from config.config import DATA_VALIDATION_CONFIG

logger = get_logger(__name__)

class DataValidation:
    def __init__(self):
        """
        Initialize data validation with configuration
        """
        self.validation_config = DATA_VALIDATION_CONFIG
        
    def read_schema(self):
        """
        Read schema configuration from file
        
        Returns:
            Dict containing schema information
        """
        try:
            with open(self.validation_config["schema_dir"], 'r') as f:
                schema = json.load(f)
            
            logger.info(f"Schema loaded successfully from {self.validation_config['schema_dir']}")
            return schema
        except Exception as e:
            logger.error(f"Error reading schema: {e}")
            raise CustomException(e)
    
    def validate_dataset_schema(self, dataframe):
        """
        Validate if the dataset matches the expected schema
        
        Args:
            dataframe: Pandas DataFrame to validate
            
        Returns:
            Boolean indicating if validation passed
        """
        try:
            schema = self.read_schema()
            validation_status = True
            
            # Check if all required columns are present
            for column in schema["columns"]:
                if column not in dataframe.columns:
                    validation_status = False
                    logger.error(f"Required column {column} not found in the dataset")
            # Special handling: if 'Wafer' column is missing but there is an unnamed first column, treat it as 'Wafer'
            if not validation_status and 'Wafer' not in dataframe.columns:
                first_col = str(dataframe.columns[0])
                if first_col.strip() == '' or first_col.lower().startswith('unnamed'):
                    dataframe.rename(columns={dataframe.columns[0]: 'Wafer'}, inplace=True)
                    logger.info("Renamed first unnamed column to 'Wafer' during validation")
                    # Re-check required columns
                    validation_status = all(col in dataframe.columns for col in schema["columns"]) and (schema["target_column"] in dataframe.columns)
            
            # Check if target column exists
            if schema["target_column"] not in dataframe.columns:
                validation_status = False
                logger.error(f"Target column {schema['target_column']} not found in the dataset")
            
            # Check domain values for target column
            if validation_status and "domain_value" in schema:
                for column, values in schema["domain_value"].items():
                    if column in dataframe.columns:
                        unique_values = dataframe[column].unique()
                        for val in unique_values:
                            if val not in values:
                                validation_status = False
                                logger.error(f"Invalid value {val} found in column {column}")
            
            return validation_status
        
        except Exception as e:
            logger.error(f"Error in dataset schema validation: {e}")
            raise CustomException(e)
    
    def initiate_data_validation(self, train_file_path, test_file_path):
        """
        Initiate the data validation process
        
        Args:
            train_file_path: Path to training data
            test_file_path: Path to test data
            
        Returns:
            Boolean indicating if validation passed
        """
        try:
            logger.info("Starting data validation")
            
            # Create validation directory
            os.makedirs(os.path.dirname(self.validation_config["report_file_path"]), exist_ok=True)
            
            # Read data
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            logger.info(f"Train dataset shape: {train_df.shape}")
            logger.info(f"Test dataset shape: {test_df.shape}")
            
            # Validate schema
            train_validation = self.validate_dataset_schema(train_df)
            test_validation = self.validate_dataset_schema(test_df)
            
            validation_status = train_validation and test_validation
            
            # Generate validation report
            validation_report = {
                "train_file_path": train_file_path,
                "test_file_path": test_file_path,
                "train_validation": train_validation,
                "test_validation": test_validation,
                "validation_status": validation_status
            }
            
            # Save validation report
            with open(self.validation_config["report_file_path"], 'w') as f:
                json.dump(validation_report, f, indent=4)
            
            logger.info(f"Data validation completed with status: {validation_status}")
            logger.info(f"Validation report saved at: {self.validation_config['report_file_path']}")
            
            return validation_status
            
        except Exception as e:
            logger.error(f"Error in data validation: {e}")
            raise CustomException(e)