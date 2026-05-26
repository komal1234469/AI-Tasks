# Serve an NLP Model using REST API with Flask

# Install required libraries
# pip install flask transformers torch

from flask import Flask, request, jsonify
from transformers import pipeline

# -----------------------------
# Step 1: Initialize Flask App
# -----------------------------

app = Flask(__name__)

# -----------------------------
# Step 2: Load NLP Model
# -----------------------------

# Sentiment Analysis Pipeline
classifier = pipeline(
    "sentiment-analysis"
)

# -----------------------------
# Step 3: Health Check Route
# -----------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "NLP Model Flask API is running"
    })

# -----------------------------
# Step 4: Prediction Route
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict_sentiment():

    # Get JSON data
    data = request.get_json()

    # Validate input
    if not data or "text" not in data:

        return jsonify({
            "error": "Please provide 'text' in request body"
        }), 400

    text = data["text"]

    # Generate prediction
    result = classifier(text)

    # Return response
    return jsonify({
        "input_text": text,
        "prediction": result
    })

# -----------------------------
# Step 5: Run Flask Server
# -----------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )

# -----------------------------
# Step 6: Example API Request
# -----------------------------

# POST http://127.0.0.1:5000/predict

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
# Step 7: Key Learning
# -----------------------------

print("✅ NLP Model Flask API Ready")
print("The Flask API can now serve NLP model predictions.")