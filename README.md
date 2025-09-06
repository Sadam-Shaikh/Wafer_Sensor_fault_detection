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

## Usage

1. Place your wafer sensor data in the `data/raw` directory
2. Run the training pipeline to train the model
3. Use the web interface to make predictions on new data
