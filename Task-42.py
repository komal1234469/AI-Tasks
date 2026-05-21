# Day 42: Improve Retrieval Accuracy in RAG Systems

# Install required libraries
# pip install langchain langchain-openai faiss-cpu openai

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------
# Step 1: Create Knowledge Base
# -----------------------------

documents = [
    Document(
        page_content="RAG combines retrieval with generation to improve factual accuracy."
    ),

    Document(
        page_content="FAISS performs efficient vector similarity search."
    ),

    Document(
        page_content="Chunking strategy affects retrieval quality in RAG systems."
    ),

    Document(
        page_content="Embeddings capture semantic meaning of text."
    ),

    Document(
        page_content="Better embeddings improve retrieval relevance."
    )
]

# -----------------------------
# Step 2: Generate Embeddings
# -----------------------------

embeddings = OpenAIEmbeddings(
    api_key="YOUR_API_KEY"
)

# -----------------------------
# Step 3: Build FAISS Vector Store
# -----------------------------

vectorstore = FAISS.from_documents(
    documents,
    embeddings
)

# -----------------------------
# Step 4: Create Optimized Retriever
# -----------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)

# -----------------------------
# Step 5: User Query
# -----------------------------

query = "How can retrieval accuracy be improved in RAG?"

# Retrieve relevant docs
retrieved_docs = retriever.invoke(query)

# -----------------------------
# Step 6: Display Retrieved Docs
# -----------------------------

print("\n🔍 Query:\n")
print(query)

print("\n📄 Retrieved Documents:\n")

for i, doc in enumerate(retrieved_docs, start=1):
    print(f"Document {i}:")
    print(doc.page_content)
    print("-" * 50)

# -----------------------------
# Step 7: Analyze Similarity Scores
# -----------------------------

query_embedding = embeddings.embed_query(query)

print("\n📊 Similarity Scores:\n")

for doc in retrieved_docs:

    doc_embedding = embeddings.embed_query(
        doc.page_content
    )

    score = cosine_similarity(
        [query_embedding],
        [doc_embedding]
    )[0][0]

    print(f"Document: {doc.page_content}")
    print(f"Similarity Score: {score:.4f}")
    print("-" * 50)

# -----------------------------
# Step 8: Retrieval Optimization Ideas
# -----------------------------

print("\n🚀 RAG Optimization Techniques:\n")

techniques = [
    "Use better embedding models",
    "Improve chunk size and chunk overlap",
    "Store cleaner and more focused documents",
    "Increase top-k retrieval carefully",
    "Apply reranking after retrieval",
    "Use metadata filtering",
    "Remove duplicate or noisy chunks"
]

for t in techniques:
    print(f"- {t}")

# -----------------------------
# Step 9: Key Learning
# -----------------------------

print("\n💡 Key Learning:")
print("Retrieval quality directly impacts RAG accuracy. Better embeddings and chunking strategies significantly reduce hallucinations.")