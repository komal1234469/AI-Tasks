# Day 51: Automate Model Deployment Pipeline
# CI/CD Pipeline for ML Model Deployment

# ==========================================
# Step 1: Train and Save Model
# File: train_model.py
# ==========================================

from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

# Sample Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([30, 40, 50, 60, 70])

# Train Model
model = LinearRegression()
model.fit(X, y)

# Save Model
joblib.dump(
    model,
    "model.pkl"
)

print("✅ Model Saved Successfully")

# ==========================================
# Step 2: FastAPI Inference App
# File: main.py
# ==========================================

from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

@app.get("/")
def home():

    return {
        "message": "ML Model API Running"
    }

@app.get("/predict")
def predict(hours: float):

    prediction = model.predict(
        [[hours]]
    )[0]

    return {
        "hours": hours,
        "prediction": round(
            float(prediction),
            2
        )
    }

# ==========================================
# Step 3: requirements.txt
# ==========================================

"""
fastapi
uvicorn
scikit-learn
joblib
"""

# ==========================================
# Step 4: Dockerfile
# ==========================================

"""
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn",
     "main:app",
     "--host",
     "0.0.0.0",
     "--port",
     "8000"]
"""

# ==========================================
# Step 5: Kubernetes Deployment
# deployment.yaml
# ==========================================

"""
apiVersion: apps/v1
kind: Deployment

metadata:
  name: ml-model-deployment

spec:
  replicas: 2

  selector:
    matchLabels:
      app: ml-model

  template:
    metadata:
      labels:
        app: ml-model

    spec:
      containers:
      - name: ml-model

        image:
          YOUR_DOCKERHUB_USERNAME/ml-model:v1

        ports:
        - containerPort: 8000
"""

# ==========================================
# Step 6: GitHub Actions CI/CD
# .github/workflows/deploy.yml
# ==========================================

"""
name: Deploy ML Model

on:
  push:
    branches:
      - main

jobs:

  build-and-deploy:

    runs-on: ubuntu-latest

    steps:

    - name: Checkout Code

      uses: actions/checkout@v4

    - name: Setup Python

      uses: actions/setup-python@v5

      with:
        python-version: '3.11'

    - name: Install Dependencies

      run: |
        pip install -r requirements.txt

    - name: Train Model

      run: |
        python train_model.py

    - name: Build Docker Image

      run: |
        docker build -t ml-model .

    - name: Login Docker Hub

      uses: docker/login-action@v3

      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Push Docker Image

      run: |
        docker tag ml-model \
        YOUR_DOCKERHUB_USERNAME/ml-model:v1

        docker push \
        YOUR_DOCKERHUB_USERNAME/ml-model:v1

    - name: Deploy to Kubernetes

      run: |
        kubectl apply -f deployment.yaml
"""

# ==========================================
# Step 7: Pipeline Flow
# ==========================================

pipeline_steps = [

    "Developer pushes code to GitHub",

    "GitHub Actions starts automatically",

    "Dependencies installed",

    "Model retrained",

    "Docker image built",

    "Image pushed to Docker Hub",

    "Kubernetes deployment updated",

    "New model available in production"
]

print("\n🚀 Automated Deployment Pipeline:\n")

for step in pipeline_steps:

    print(f"✔ {step}")

# ==========================================
# Step 8: Key Learning
# ==========================================

print("\n💡 Key Learning:")
print(
    "CI/CD pipelines automate training, packaging, testing, and deployment of ML models, reducing manual effort and deployment errors."
)