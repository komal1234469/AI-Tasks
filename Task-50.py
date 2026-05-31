# Day 50: Deploy Dockerized FastAPI App on Kubernetes Cluster
# Complete Project

# ==========================================
# File: main.py
# ==========================================

from fastapi import FastAPI

app = FastAPI(
    title="Kubernetes Demo API",
    version="1.0"
)

@app.get("/")
def home():

    return {
        "message": "Hello from Kubernetes Cluster 🚀"
    }

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }

@app.get("/predict")
def predict(hours: float):

    predicted_marks = hours * 10 + 25

    return {
        "hours_studied": hours,
        "predicted_marks": predicted_marks
    }


# ==========================================
# File: requirements.txt
# ==========================================

"""
fastapi
uvicorn
"""

# ==========================================
# File: Dockerfile
# ==========================================

"""
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
"""

# ==========================================
# File: deployment.yaml
# ==========================================

"""
apiVersion: apps/v1
kind: Deployment

metadata:
  name: fastapi-deployment

spec:
  replicas: 3

  selector:
    matchLabels:
      app: fastapi-app

  template:
    metadata:
      labels:
        app: fastapi-app

    spec:
      containers:

      - name: fastapi-container

        image: YOUR_DOCKERHUB_USERNAME/fastapi-k8s-app:v1

        ports:
        - containerPort: 8000
"""

# ==========================================
# File: service.yaml
# ==========================================

"""
apiVersion: v1
kind: Service

metadata:
  name: fastapi-service

spec:
  selector:
    app: fastapi-app

  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000

  type: LoadBalancer
"""

# ==========================================
# Build Docker Image
# ==========================================

"""
docker build -t fastapi-k8s-app .
"""

# ==========================================
# Run Docker Container
# ==========================================

"""
docker run -p 8000:8000 fastapi-k8s-app
"""

# ==========================================
# Push to Docker Hub
# ==========================================

"""
docker login

docker tag fastapi-k8s-app YOUR_DOCKERHUB_USERNAME/fastapi-k8s-app:v1

docker push YOUR_DOCKERHUB_USERNAME/fastapi-k8s-app:v1
"""

# ==========================================
# Deploy to Kubernetes
# ==========================================

"""
kubectl apply -f deployment.yaml

kubectl apply -f service.yaml
"""

# ==========================================
# Verify Deployment
# ==========================================

"""
kubectl get deployments

kubectl get pods

kubectl get services
"""

# ==========================================
# Scale Application
# ==========================================

"""
kubectl scale deployment fastapi-deployment --replicas=5
"""

# ==========================================
# View Logs
# ==========================================

"""
kubectl logs <pod-name>
"""

# ==========================================
# Delete Deployment
# ==========================================

"""
kubectl delete -f deployment.yaml

kubectl delete -f service.yaml
"""

# ==========================================
# Project Structure
# ==========================================

"""
project/
│
├── main.py
├── requirements.txt
├── Dockerfile
├── deployment.yaml
└── service.yaml
"""

print("✅ FastAPI App Ready for Docker & Kubernetes Deployment")