# Wafer Sensor Fault Detection

This project uses machine learning to predict whether a wafer is good or bad based on sensor data.

## Project Structure
```
├── artifacts/             # Trained models and data files
├── data/                  # Data directory
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── logs/                  # Log files
├── src/                   # Source code
│   ├── components/        # Components for each stage of ML pipeline
│   ├── pipeline/          # Pipeline modules
│   ├── utils/             # Utility functions
│   └── app.py             # Flask web application
├── config/                # Configuration files
├── tests/                 # Test files
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Setup Instructions

1. Create a virtual environment:
```
python -m venv venv
```

2. Activate the virtual environment:
```
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Run the application:
```
python src/app.py
```

## Environment Variables

This project uses environment variables for configuration. Create a `.env` file in the root directory with the following variables:

```
PYTHON_VERSION=3.11.9
PYTHONUNBUFFERED=1
PIP_ONLY_BINARY=:all:
DATA_FILE_URL=https://your-public-bucket/path/to/your_labeled_dataset.csv
PYTHONPATH=.
```

These variables will be automatically loaded when the application starts.

## Usage

1. Place your wafer sensor data in the `data/raw` directory or specify a URL in the `.env` file
2. Run the training pipeline to train the model
3. Use the web interface to make predictions on new data
