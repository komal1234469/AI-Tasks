# Day 38: Store and Retrieve Vectors using FAISS

# Install required libraries
# pip install faiss-cpu sentence-transformers numpy

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Sample documents
documents = [
    "Artificial Intelligence is transforming industries.",
    "Machine learning learns patterns from data.",
    "FAISS enables fast vector similarity search.",
    "Embeddings convert text into numerical vectors.",
    "Semantic search understands contextual meaning."
]

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(documents)

# Convert embeddings to float32
vectors = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = vectors.shape[1]

index = faiss.IndexFlatL2(dimension)

# Store vectors in FAISS
index.add(vectors)

print("🔹 Total vectors stored:", index.ntotal)

# User query
query = "How do vector databases perform semantic search?"

# Generate query embedding
query_vector = model.encode([query]).astype("float32")

# Retrieve top 3 similar vectors
k = 3

distances, indices = index.search(query_vector, k)

# Print results
print("\n🔹 Query:\n")
print(query)

print("\n🔹 Top Matching Documents:\n")

for i, idx in enumerate(indices[0]):
    print(f"Result {i+1}:")
    print(documents[idx])
    print(f"Distance Score: {distances[0][i]:.4f}")
    print("-" * 50)