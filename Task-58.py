# Day 54: RAG Optimization

import redis
import time

# Redis Cache
cache = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

def get_answer(question):

    # Step 1: Check Cache
    cached = cache.get(question)

    if cached:
        return {
            "source": "cache",
            "answer": cached
        }

    start = time.time()

    # Step 2: Retrieve Top Documents
    docs = vector_db.similarity_search(
        question,
        k=2   # reduced from 5
    )

    # Step 3: Build Smaller Context
    context = "\n".join(
        doc.page_content[:300]
        for doc in docs
    )

    # Step 4: Optimized Prompt
    prompt = f"""
Answer using only the context.

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    answer = response.content

    # Step 5: Store in Cache
    cache.setex(
        question,
        3600,
        answer
    )

    latency = time.time() - start

    return {
        "answer": answer,
        "latency": round(latency, 2),
        "source": "llm"
    }