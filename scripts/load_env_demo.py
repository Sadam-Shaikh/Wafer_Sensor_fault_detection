import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Access environment variables
    python_version = os.getenv("PYTHON_VERSION")
    data_file_url = os.getenv("DATA_FILE_URL")
    
    logger.info(f"Python Version: {python_version}")
    logger.info(f"Data File URL: {data_file_url}")
    
    # Check if data URL is available
    if data_file_url:
        logger.info("Data URL is available. You can use this to download the dataset.")
        # Here you could call the data ingestion process
        # from src.components.data_ingestion import DataIngestion
        # data_ingestion = DataIngestion()
        # data_ingestion.initiate_data_ingestion()
    else:
        logger.info("No Data URL provided in environment variables.")
        logger.info("Using local files from data/raw directory.")

if __name__ == "__main__":
    main()