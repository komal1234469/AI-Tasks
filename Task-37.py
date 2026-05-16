# Day 37: Generate Embeddings for Documents

# Install required libraries
# pip install openai

from openai import OpenAI
import numpy as np

# Initialize OpenAI client
client = OpenAI(
    api_key="YOUR_API_KEY"   # Replace with your API key
)

# Sample documents
documents = [
    "Artificial Intelligence is transforming industries.",
    "Machine learning helps systems learn from data.",
    "Embeddings convert text into numerical vectors.",
    "Semantic search retrieves information using meaning.",
    "Chatbots use NLP to communicate with users."
]

print("🔹 Generating Embeddings...\n")

# Generate embeddings
embeddings = []

for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )

    embedding = response.data[0].embedding
    embeddings.append(embedding)

    print(f"Document: {doc}")
    print(f"Embedding Length: {len(embedding)}")
    print("-" * 50)

# Convert to NumPy array
embedding_matrix = np.array(embeddings)

print("\n🔹 Final Embedding Matrix Shape:")
print(embedding_matrix.shape)