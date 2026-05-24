# Day 45: Create an API to Serve Model Predictions

# Install required libraries
# pip install fastapi uvicorn scikit-learn pandas

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import numpy as np

# -----------------------------
# Step 1: Create Sample Dataset
# -----------------------------

# Hours studied vs marks scored

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([
    35,
    45,
    55,
    65,
    75
])

# -----------------------------
# Step 2: Train ML Model
# -----------------------------

model = LinearRegression()

model.fit(X, y)

# -----------------------------
# Step 3: Initialize FastAPI
# -----------------------------

app = FastAPI(
    title="ML Prediction API",
    version="1.0"
)

# -----------------------------
# Step 4: Define Request Schema
# -----------------------------

class PredictionRequest(BaseModel):
    hours: float

# -----------------------------
# Step 5: Health Check Endpoint
# -----------------------------

@app.get("/")

def home():

    return {
        "message": "ML Prediction API is running"
    }

# -----------------------------
# Step 6: Prediction Endpoint
# -----------------------------

@app.post("/predict")

def predict(request: PredictionRequest):

    # Convert input to model format
    input_data = np.array([
        [request.hours]
    ])

    # Generate prediction
    prediction = model.predict(input_data)[0]

    return {
        "hours_studied": request.hours,
        "predicted_marks": round(prediction, 2)
    }

# -----------------------------
# Step 7: Run Server
# -----------------------------

# Run using:
# uvicorn filename:app --reload

# Example:
# uvicorn main:app --reload

# -----------------------------
# Step 8: Example API Request
# -----------------------------

# POST http://127.0.0.1:8000/predict

# JSON Input:
# {
#   "hours": 6
# }

# Example Output:
# {
#   "hours_studied": 6,
#   "predicted_marks": 85.0
# }

# -----------------------------
# Step 9: Key Learning
# -----------------------------

print("✅ FastAPI Prediction Service Ready")
print("The API can now serve machine learning model predictions.")