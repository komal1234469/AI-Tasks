# Day 40: Store Document Metadata and Embeddings in MongoDB

# Install required libraries
# pip install pymongo sentence-transformers numpy

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# Step 1: Connect to MongoDB
# -----------------------------

client = MongoClient("mongodb://localhost:27017/")

db = client["vector_database"]

collection = db["documents"]

# -----------------------------
# Step 2: Load Embedding Model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Step 3: Sample Documents
# -----------------------------

documents = [
    {
        "title": "Machine Learning Basics",
        "category": "AI",
        "content": "Machine learning allows systems to learn from data."
    },
    {
        "title": "MongoDB Search",
        "category": "Database",
        "content": "MongoDB can store metadata and vector embeddings."
    },
    {
        "title": "RAG Systems",
        "category": "Generative AI",
        "content": "RAG combines retrieval with generation for better accuracy."
    }
]

# -----------------------------
# Step 4: Generate Embeddings
# -----------------------------

for doc in documents:

    embedding = model.encode(doc["content"]).tolist()

    mongo_document = {
        "title": doc["title"],
        "category": doc["category"],
        "content": doc["content"],
        "embedding": embedding
    }

    collection.insert_one(mongo_document)

print("✅ Documents with embeddings stored successfully in MongoDB.")

# -----------------------------
# Step 5: Query Similar Documents
# -----------------------------

query = "How are embeddings stored in databases?"

query_embedding = model.encode(query)

# Fetch all stored docs
stored_docs = list(collection.find())

# -----------------------------
# Step 6: Compute Similarity
# -----------------------------

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

similarities = []

for doc in stored_docs:

    score = cosine_similarity(
        query_embedding,
        doc["embedding"]
    )

    similarities.append((doc["title"], score))

# Sort by highest similarity
similarities.sort(key=lambda x: x[1], reverse=True)

# -----------------------------
# Step 7: Output Results
# -----------------------------

print("\n🔍 Query:")
print(query)

print("\n📄 Most Similar Documents:\n")

for title, score in similarities:
    print(f"Title: {title}")
    print(f"Similarity Score: {score:.4f}")
    print("-" * 50)

# -----------------------------
# Step 8: Key Learning
# -----------------------------

print("\n💡 Key Learning:")
print("MongoDB can store both structured metadata and vector embeddings, enabling semantic search and retrieval systems.")