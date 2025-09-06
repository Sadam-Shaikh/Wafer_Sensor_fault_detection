import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

# Create timestamp for versioning
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

# Training pipeline config
TRAINING_PIPELINE_CONFIG = {
    "pipeline_name": "wafer_fault_detection",
    "artifact_dir": os.path.join(ARTIFACTS_DIR, CURRENT_TIME_STAMP)
}

# Data ingestion config
DATA_INGESTION_CONFIG = {
    "dataset_download_url": os.getenv("DATA_FILE_URL"),  # Get URL from environment variable
    "raw_data_dir": RAW_DATA_DIR,
    "ingested_dir": os.path.join(ARTIFACTS_DIR, CURRENT_TIME_STAMP, "data_ingestion"),
    "ingested_train_dir": os.path.join(ARTIFACTS_DIR, CURRENT_TIME_STAMP, "data_ingestion", "train"),
    "ingested_test_dir": os.path.join(ARTIFACTS_DIR, CURRENT_TIME_STAMP, "data_ingestion", "test"),
}

# Data validation config
DATA_VALIDATION_CONFIG = {
    "schema_dir": os.path.join(CONFIG_DIR, "schema.json"),
    "validation_dir": os.path.join(ARTIFACTS_DIR, CURRENT_TIME_STAMP, "data_validation"),
    "report_file_path": os.path.join(ARTIFACTS_DIR, CURRENT_TIME_STAMP, "data_validation", "report.json"),
}

# Data transformation config
DATA_TRANSFORMATION_CONFIG = {
    "transformed_dir": os.path.join(ARTIFACTS_DIR, CURRENT_TIME_STAMP, "data_transformation"),
    "transformed_train_dir": os.path.join(ARTIFACTS_DIR, CURRENT_TIME_STAMP, "data_transformation", "train"),
    "transformed_test_dir": os.path.join(ARTIFACTS_DIR, CURRENT_TIME_STAMP, "data_transformation", "test"),
    "preprocessed_object_file_path": os.path.join(ARTIFACTS_DIR, CURRENT_TIME_STAMP, "data_transformation", "preprocessed.pkl"),
}

# Model training config
MODEL_TRAINER_CONFIG = {
    "trained_model_dir": os.path.join(ARTIFACTS_DIR, CURRENT_TIME_STAMP, "model_trainer"),
    "model_file_path": os.path.join(ARTIFACTS_DIR, CURRENT_TIME_STAMP, "model_trainer", "model.pkl"),
    "base_accuracy": 0.3,  # Lowered for synthetic data
}

# Model evaluation config
MODEL_EVALUATION_CONFIG = {
    "model_evaluation_file_path": os.path.join(ARTIFACTS_DIR, CURRENT_TIME_STAMP, "model_evaluation", "metrics.json"),
    "threshold": 0.7,
}

# Model pusher config
MODEL_PUSHER_CONFIG = {
    "model_export_dir": os.path.join(ARTIFACTS_DIR, "exported_models"),
}