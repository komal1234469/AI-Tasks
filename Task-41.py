# Day 41: Build Retrieval-Augmented Generation (RAG) Pipeline

# Install required libraries
# pip install langchain langchain-openai faiss-cpu openai

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# -----------------------------
# Step 1: Sample Knowledge Base
# -----------------------------

documents = [
    Document(
        page_content="Machine learning learns patterns from data."
    ),

    Document(
        page_content="RAG combines retrieval with generation."
    ),

    Document(
        page_content="FAISS enables fast vector similarity search."
    ),

    Document(
        page_content="Embeddings convert text into numerical vectors."
    )
]

# -----------------------------
# Step 2: Create Embeddings
# -----------------------------

embeddings = OpenAIEmbeddings(
    api_key="YOUR_API_KEY"
)

# -----------------------------
# Step 3: Build Vector Store
# -----------------------------

vectorstore = FAISS.from_documents(
    documents,
    embeddings
)

# -----------------------------
# Step 4: Create Retriever
# -----------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# -----------------------------
# Step 5: User Query
# -----------------------------

query = "How does RAG reduce hallucination?"

# Retrieve relevant documents
retrieved_docs = retriever.invoke(query)

# -----------------------------
# Step 6: Display Retrieved Docs
# -----------------------------

print("\n🔍 User Query:\n")
print(query)

print("\n📄 Retrieved Documents:\n")

for i, doc in enumerate(retrieved_docs, start=1):
    print(f"Document {i}:")
    print(doc.page_content)
    print("-" * 50)

# -----------------------------
# Step 7: Build Prompt
# -----------------------------

context = "\n".join(
    [doc.page_content for doc in retrieved_docs]
)

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

If the answer is not available in the context,
say "Information not found in context."
"""

# -----------------------------
# Step 8: Generate Answer
# -----------------------------

llm = ChatOpenAI(
    api_key="YOUR_API_KEY",
    model="gpt-3.5-turbo"
)

response = llm.invoke(prompt)

# -----------------------------
# Step 9: Output Final Response
# -----------------------------

print("\n🤖 AI Response:\n")
print(response.content)

# -----------------------------
# Step 10: Key Learning
# -----------------------------

print("\n💡 Key Learning:")
print("RAG improves answer quality by retrieving relevant external knowledge before generation.")