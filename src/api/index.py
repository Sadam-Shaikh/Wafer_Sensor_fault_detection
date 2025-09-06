import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from flask import Flask, request, jsonify
from src.pipeline.predict_pipeline import PredictionPipeline, CustomData
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "message": "Wafer Sensor Fault Detection API",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Make predictions with CSV data",
            "/health": "GET - Check API health"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if request has JSON data
        if request.is_json:
            data = request.get_json()
            
            # Convert JSON to DataFrame
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                df = pd.DataFrame(data['data'])
            else:
                return jsonify({"error": "Invalid JSON format. Expected array of objects or {data: [array of objects]}"}), 400
        
        # If no JSON, check for file upload
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Check file extension
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            else:
                return jsonify({"error": "Unsupported file format. Please upload CSV or Excel file."}), 400
        else:
            return jsonify({"error": "No data provided. Send JSON data or upload a file."}), 400
        
        # Make predictions
        prediction_pipeline = PredictionPipeline()
        predictions = prediction_pipeline.predict(df)
        
        # Format response
        result = {
            "predictions": predictions.tolist(),
            "prediction_labels": ["Bad Wafer" if pred == 0 else "Good Wafer" for pred in predictions],
            "summary": {
                "total": len(predictions),
                "good_wafers": int((predictions == 1).sum()),
                "bad_wafers": int((predictions == 0).sum())
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)