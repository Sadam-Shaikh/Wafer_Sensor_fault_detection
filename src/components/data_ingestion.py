import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.logger import get_logger
from utils.exception import CustomException
import config.config as config

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        """
        Initialize data ingestion with configuration
        """
        self.ingestion_config = config.DATA_INGESTION_CONFIG
        
    def initiate_data_ingestion(self, file_path=None):
        """
        Initiate the data ingestion process
        
        Args:
            file_path: Path to the input data file
            
        Returns:
            Tuple of paths to train and test data
        """
        try:
            # Create directories
            os.makedirs(self.ingestion_config["ingested_train_dir"], exist_ok=True)
            os.makedirs(self.ingestion_config["ingested_test_dir"], exist_ok=True)
            
            logger.info("Data ingestion initiated")
            
            # If file path is not provided, pick a valid wafer csv from raw data directory
            if file_path is None:
                raw_dir = self.ingestion_config["raw_data_dir"]
                raw_files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.csv')]
                if len(raw_files) == 0:
                    raise Exception("No CSV files found in raw data directory")
                # Prefer files that look like wafer datasets
                wafer_like = [f for f in raw_files if 'wafer' in f.lower()]
                chosen = wafer_like[0] if wafer_like else raw_files[0]
                file_path = os.path.join(raw_dir, chosen)
            
            logger.info(f"Reading data from {file_path}")
            
            # Read data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                # Many wafer CSVs have an unnamed first column that contains Wafer IDs
                first_col = str(df.columns[0])
                if first_col.strip() == '' or first_col.lower().startswith('unnamed'):
                    df = df.rename(columns={df.columns[0]: 'Wafer'})
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                raise Exception(f"Unsupported file format: {file_path}")
            
            logger.info(f"Dataset shape: {df.shape}")

            # Normalize target labels to schema domain {0, 1}
            target_col = "Good/Bad"
            if target_col in df.columns:
                df[target_col] = df[target_col].replace({-1: 0, 1: 1}).astype(int)
            
            # Split data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save train and test data
            train_file_path = os.path.join(self.ingestion_config["ingested_train_dir"], "train.csv")
            test_file_path = os.path.join(self.ingestion_config["ingested_test_dir"], "test.csv")
            
            train_set.to_csv(train_file_path, index=False)
            test_set.to_csv(test_file_path, index=False)
            
            logger.info(f"Train data saved at: {train_file_path}")
            logger.info(f"Test data saved at: {test_file_path}")
            
            return train_file_path, test_file_path
            
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            raise CustomException(e)