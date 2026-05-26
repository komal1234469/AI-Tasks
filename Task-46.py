# Serve an NLP Model using REST API with FastAPI

# Install required libraries
# pip install fastapi uvicorn transformers torch

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

# -----------------------------
# Step 1: Initialize FastAPI
# -----------------------------

app = FastAPI(
    title="NLP REST API",
    version="1.0"
)

# -----------------------------
# Step 2: Load NLP Model
# -----------------------------

# Sentiment Analysis Pipeline
classifier = pipeline(
    "sentiment-analysis"
)

# -----------------------------
# Step 3: Request Schema
# -----------------------------

class TextRequest(BaseModel):
    text: str

# -----------------------------
# Step 4: Health Check Endpoint
# -----------------------------

@app.get("/")

def home():

    return {
        "message": "NLP Model API is running"
    }

# -----------------------------
# Step 5: Prediction Endpoint
# -----------------------------

@app.post("/predict")

def predict_sentiment(request: TextRequest):

    text = request.text

    # Model Prediction
    result = classifier(text)

    return {
        "input_text": text,
        "prediction": result
    }

# -----------------------------
# Step 6: Run Server
# -----------------------------

# Run using:
# uvicorn main:app --reload

# Example:
# uvicorn main:app --reload

# -----------------------------
# Step 7: Example API Request
# -----------------------------

# POST http://127.0.0.1:8000/predict

# JSON Input:
# {
#   "text": "I love learning AI and NLP!"
# }

# Example Output:
# {
#   "input_text": "I love learning AI and NLP!",
#   "prediction": [
#       {
#           "label": "POSITIVE",
#           "score": 0.9998
#       }
#   ]
# }

# -----------------------------
# Step 8: Key Learning
# -----------------------------

print("✅ NLP Model REST API Ready")
print("The API can now serve NLP predictions using FastAPI.")