# ==============================
# 📦 1. Import Libraries
# ==============================
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ==============================
# 📄 2. Sample Documents
# ==============================
documents = [
    "Cyber crime involves hacking and phishing attacks",
    "FAISS is used for similarity search in vector databases",
    "Wireshark helps in network packet analysis",
    "Trojans are malicious software disguised as legitimate apps",
    "Digital signatures ensure data integrity and authentication",
    "Machine learning models improve with hyperparameter tuning",
    "Neural networks learn patterns from large datasets"
]

# ==============================
# 🤖 3. Load Embedding Model
# ==============================
model = SentenceTransformer('all-MiniLM-L6-v2')

# ==============================
# 🔢 4. Convert Documents to Embeddings
# ==============================
doc_embeddings = model.encode(documents)
doc_embeddings = np.array(doc_embeddings).astype('float32')

# ==============================
# 🏗️ 5. Build FAISS Index
# ==============================
dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
index.add(doc_embeddings)

print("✅ FAISS index built successfully!")

# ==============================
# 🔍 6. Function for Semantic Search
# ==============================
def search(query, top_k=3):
    # Convert query to embedding
    query_vec = model.encode([query]).astype('float32')

    # Search in FAISS index
    distances, indices = index.search(query_vec, top_k)

    print("\n🔎 Query:", query)
    print("\n📌 Top Results:\n")

    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {documents[idx]} (Distance: {distances[0][i]:.4f})")

# ==============================
# 🚀 7. Test the Search
# ==============================
search("What is malware like trojan?")
search("How does network security work?")
search("AI and machine learning improvements")