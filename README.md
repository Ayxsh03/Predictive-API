# Predictive Analysis API

This API predicts machine downtime using manufacturing data. It provides endpoints to upload data, train a machine learning model, and make predictions.

---

## Setup Instructions

### Prerequisites
1. Install Python (>= 3.7).
2. Install the required libraries using pip:
   ```bash
   pip install flask scikit-learn pandas numpy joblib
   ```

### Running the API
1. Save the code in a file named `app.py`.
2. Run the Flask server:
   ```bash
   python app.py
   ```
   By default, the server runs on `http://127.0.0.1:5000/`.

---

## Endpoints

### 1. **Upload Data**
   **Endpoint:** `POST /upload`

   **Description:** Upload a CSV file containing manufacturing data.

   **Request:**
   - Form Data: `file` (CSV file with columns like `Machine_ID`, `Temperature`, `Run_Time`, `Downtime_Flag`)

   **Response:**
   ```json
   {
       "message": "File uploaded successfully"
   }
   ```

### 2. **Train Model**
   **Endpoint:** `POST /train`

   **Description:** Train a decision tree model using the uploaded data.

   **Response:**
   ```json
   {
       "accuracy": 0.85,
       "f1_score": 0.80
   }
   ```

### 3. **Predict Downtime**
   **Endpoint:** `POST /predict`

   **Description:** Predict whether a machine will have downtime based on input parameters.

   **Request:**
   ```json
   {
       "Temperature": 80,
       "Run_Time": 120
   }
   ```

   **Response:**
   ```json
   {
       "Downtime": "Yes",
       "Confidence": 0.85
   }
   ```

---

## Example API Usage

### 1. Upload Data
   **Command:**
   ```bash
   curl -X POST -F "file=@data.csv" http://127.0.0.1:5000/upload
   ```
   **Response:**
   ```json
   {
       "message": "File uploaded successfully"
   }
   ```

### 2. Train Model
   **Command:**
   ```bash
   curl -X POST http://127.0.0.1:5000/train
   ```
   **Response:**
   ```json
   {
       "accuracy": 0.85,
       "f1_score": 0.80
   }
   ```

### 3. Predict Downtime
   **Command:**
   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"Temperature": 80, "Run_Time": 120}' http://127.0.0.1:5000/predict
   ```
   **Response:**
   ```json
   {
       "Downtime": "Yes",
       "Confidence": 0.85
   }
   ```

---

## Notes
- If no dataset is uploaded, synthetic data is automatically generated for testing.
- Use tools like Postman for easier testing of endpoints.
- Ensure all required columns are present in the dataset: `Machine_ID`, `Temperature`, `Run_Time`, `Downtime_Flag`.

