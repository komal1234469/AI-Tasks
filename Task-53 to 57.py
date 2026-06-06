# Day 53: End-to-End GenAI RAG System

# Install Libraries
# pip install fastapi uvicorn
# pip install langchain langchain-openai
# pip install faiss-cpu pypdf python-multipart

from fastapi import FastAPI, UploadFile, File
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os

# ------------------------------------
# FastAPI App
# ------------------------------------

app = FastAPI(
    title="GenAI Research Assistant"
)

# ------------------------------------
# OpenAI Models
# ------------------------------------

embeddings = OpenAIEmbeddings(
    api_key="YOUR_API_KEY"
)

llm = ChatOpenAI(
    api_key="YOUR_API_KEY",
    model="gpt-4o-mini"
)

VECTOR_DB = None

# ------------------------------------
# Upload Document Endpoint
# ------------------------------------

@app.post("/upload")

async def upload_document(
    file: UploadFile = File(...)
):

    global VECTOR_DB

    file_path = file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Load PDF
    loader = PyPDFLoader(file_path)

    docs = loader.load()

    # Split Text
    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
    )

    chunks = splitter.split_documents(
        docs
    )

    # Create Vector Store
    VECTOR_DB = FAISS.from_documents(
        chunks,
        embeddings
    )

    return {
        "message":
        "Document uploaded successfully",

        "chunks":
        len(chunks)
    }

# ------------------------------------
# Ask Question Endpoint
# ------------------------------------

@app.post("/ask")

async def ask_question(
    question: str
):

    global VECTOR_DB

    if VECTOR_DB is None:

        return {
            "error":
            "Upload document first"
        }

    docs = VECTOR_DB.similarity_search(
        question,
        k=3
    )

    context = "\n".join(
        [
            doc.page_content
            for doc in docs
        ]
    )

    prompt = f"""
Answer ONLY from context.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    return {

        "question":
        question,

        "answer":
        response.content,

        "sources":
        [
            doc.page_content[:150]
            for doc in docs
        ]
    }

# ------------------------------------
# Health Check
# ------------------------------------

@app.get("/")

def home():

    return {
        "message":
        "RAG System Running"
    }

# Run:
# uvicorn main:app --reload