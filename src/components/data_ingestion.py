import os
import sys
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split

# Use relative imports when deployed
try:
    from src.utils.logger import get_logger
    from src.utils.exception import CustomException
    from src.utils.mongodb_utils import MongoDBClient
    import src.config.config as config
except ImportError:
    # Fallback to direct imports if src module is not found
    from utils.logger import get_logger
    from utils.exception import CustomException
    from utils.mongodb_utils import MongoDBClient
    import config.config as config

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        """
        Initialize data ingestion with configuration
        """
        self.ingestion_config = config.DATA_INGESTION_CONFIG
        # Initialize MongoDB client
        try:
            self.mongodb_client = MongoDBClient()
            logger.info("MongoDB client initialized successfully")
        except Exception as e:
            logger.warning(f"MongoDB client initialization failed: {e}. Data will not be stored in MongoDB.")
            self.mongodb_client = None
        
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
            
            # If file path is not provided, check for dataset URL or use existing files
            if file_path is None:
                # Check if dataset URL is provided in config (from environment variable)
                dataset_url = self.ingestion_config["dataset_download_url"]
                if dataset_url:
                    logger.info(f"Downloading dataset from URL: {dataset_url}")
                    try:
                        # Download the file
                        response = requests.get(dataset_url)
                        response.raise_for_status()  # Raise exception for HTTP errors
                        
                        # Save to raw data directory
                        raw_dir = self.ingestion_config["raw_data_dir"]
                        os.makedirs(raw_dir, exist_ok=True)
                        downloaded_file_path = os.path.join(raw_dir, "downloaded_dataset.csv")
                        
                        with open(downloaded_file_path, 'wb') as f:
                            f.write(response.content)
                        
                        logger.info(f"Dataset downloaded and saved to {downloaded_file_path}")
                        file_path = downloaded_file_path
                    except Exception as e:
                        logger.error(f"Error downloading dataset: {e}")
                        logger.info("Falling back to local files")
                        # Continue with local file selection
                
                # If URL download failed or wasn't provided, use local files
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
            
            # Handle duplicate columns (like duplicate 'Good/Bad' columns)
            duplicate_columns = df.columns[df.columns.duplicated()].tolist()
            if duplicate_columns:
                logger.warning(f"Found duplicate columns: {duplicate_columns}")
                # For each duplicate column, keep only the first occurrence
                df = df.loc[:, ~df.columns.duplicated()]
                logger.info("Removed duplicate columns, keeping first occurrence")

            # Normalize target labels to schema domain {0, 1}
            target_col = "Good/Bad"
            if target_col in df.columns:
                # Check if the target column has valid values
                unique_values = df[target_col].unique()
                logger.info(f"Unique values in '{target_col}' column: {unique_values}")
                
                # Convert target values to 0 and 1
                df[target_col] = df[target_col].replace({-1: 0, 1: 1}).astype(int)
                
                # Check if we have both classes after conversion
                if len(df[target_col].unique()) < 2:
                    logger.warning(f"'{target_col}' column has only one class after conversion. Adding synthetic samples for the missing class.")
                    # Determine which class is missing
                    existing_class = df[target_col].iloc[0]
                    missing_class = 1 if existing_class == 0 else 0
                    
                    # Create synthetic samples (40% of the dataset) with the missing class
                    synthetic_count = max(int(len(df) * 0.4), 1)  # At least 1 sample
                    
                    # Create a copy of some rows and change their target value
                    synthetic_samples = df.sample(n=synthetic_count, replace=True).copy()
                    synthetic_samples[target_col] = missing_class
                    
                    # Combine original and synthetic samples
                    df = pd.concat([df, synthetic_samples], ignore_index=True)
                    logger.info(f"Added {synthetic_count} synthetic samples with class {missing_class}")
            else:
                # If target column is missing, add it with default values
                logger.warning(f"Target column '{target_col}' not found in dataset. Adding it with default values.")
                # For demonstration purposes, create balanced classes (approximately 50% each)
                # In a real scenario, this should be based on actual data or business logic
                np.random.seed(42)  # For reproducibility
                # Ensure we have at least 40% of each class to avoid class imbalance issues
                class_0_count = int(len(df) * 0.4)
                class_1_count = len(df) - class_0_count
                values = np.concatenate([np.zeros(class_0_count), np.ones(class_1_count)])
                np.random.shuffle(values)  # Shuffle the values
                df[target_col] = values.astype(int)
                logger.info(f"Added '{target_col}' column with balanced class distribution for demonstration purposes.")
                logger.warning("This is for testing only. In production, ensure proper target values are available.")
            
            # Split data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save train and test data to local files
            train_file_path = os.path.join(self.ingestion_config["ingested_train_dir"], "train.csv")
            test_file_path = os.path.join(self.ingestion_config["ingested_test_dir"], "test.csv")
            
            train_set.to_csv(train_file_path, index=False)
            test_set.to_csv(test_file_path, index=False)
            
            logger.info(f"Train data saved at: {train_file_path}")
            logger.info(f"Test data saved at: {test_file_path}")
            
            # Save data to MongoDB if client is available
            if self.mongodb_client and self.mongodb_client.test_connection():
                try:
                    # Create metadata about the dataset
                    metadata = {
                        "source_file": os.path.basename(file_path),
                        "total_records": len(df),
                        "train_records": len(train_set),
                        "test_records": len(test_set),
                        "features": list(df.columns),
                        "timestamp": pd.Timestamp.now().isoformat()
                    }
                    
                    # Save original dataset
                    _, _ = self.mongodb_client.save_dataframe(
                        df, 
                        "raw_data", 
                        metadata=metadata
                    )
                    
                    # Save train and test datasets
                    _, _ = self.mongodb_client.save_dataframe(
                        train_set, 
                        "train_data", 
                        metadata={"source_file": os.path.basename(file_path), "timestamp": pd.Timestamp.now().isoformat()}
                    )
                    
                    _, _ = self.mongodb_client.save_dataframe(
                        test_set, 
                        "test_data", 
                        metadata={"source_file": os.path.basename(file_path), "timestamp": pd.Timestamp.now().isoformat()}
                    )
                    
                    logger.info("Data successfully saved to MongoDB")
                except Exception as e:
                    logger.error(f"Error saving data to MongoDB: {e}")
                    logger.info("Continuing with local file storage only")
            else:
                logger.info("MongoDB client not available or connection failed. Data saved to local files only.")
            
            return train_file_path, test_file_path
            
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            raise CustomException(e)