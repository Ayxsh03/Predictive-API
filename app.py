# Required Libraries
import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Flask App Initialization
app = Flask(__name__)

# File Names
MODEL_FILE = "model.pkl"
DATA_FILE = "synthetic_manufacturing_data.csv"

# Synthetic Data Generation (if needed)
def generate_synthetic_data(file_name):
    np.random.seed(42)
    data = pd.DataFrame({
        'Machine_ID': np.arange(1, 101),
        'Temperature': np.random.uniform(50, 100, 100),
        'Run_Time': np.random.uniform(100, 500, 100),
        'Downtime_Flag': np.random.choice([0, 1], size=100)
    })
    data.to_csv(file_name, index=False)
    return data

# Upload Endpoint
@app.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    file.save(DATA_FILE)
    return jsonify({"message": "File uploaded successfully"}), 200

# Train Endpoint
@app.route('/train', methods=['POST'])
def train_model():
    if not os.path.exists(DATA_FILE):
        return jsonify({"error": "No data uploaded"}), 400
    
    # Load and preprocess data
    data = pd.read_csv(DATA_FILE)
    X = data[['Temperature', 'Run_Time']]
    y = data['Downtime_Flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Save the model
    joblib.dump(model, MODEL_FILE)
    return jsonify({"accuracy": accuracy, "f1_score": f1}), 200

# Predict Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_FILE):
        return jsonify({"error": "Model not trained"}), 400
    
    # Parse input JSON
    data = request.get_json()
    if not all(k in data for k in ['Temperature', 'Run_Time']):
        return jsonify({"error": "Invalid input format"}), 400
    
    # Load the model
    model = joblib.load(MODEL_FILE)
    
    # Make prediction
    X = np.array([[data['Temperature'], data['Run_Time']]])
    prediction = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])
    return jsonify({"Downtime": "Yes" if prediction == 1 else "No", "Confidence": round(confidence, 2)}), 200

# Main Function
if __name__ == '__main__':
    if not os.path.exists(DATA_FILE):
        generate_synthetic_data(DATA_FILE)  # Generate synthetic data initially
    app.run(debug=True)