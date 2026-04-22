import numpy as np

# ------------------------------
# 1. Sample Text Data
# ------------------------------
texts = [
    "i love machine learning",
    "machine learning is amazing",
    "i love coding",
    "coding is fun"
]

# ------------------------------
# 2. Build Vocabulary
# ------------------------------
def build_vocab(texts):
    vocab = {}
    idx = 0
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

vocab = build_vocab(texts)
vocab_size = len(vocab)
embedding_dim = 8

# ------------------------------
# 3. Create Random Word Embeddings
# ------------------------------
np.random.seed(42)
word_embeddings = np.random.rand(vocab_size, embedding_dim)

# ------------------------------
# 4. Convert Text → Sentence Embedding (Mean Pooling)
# ------------------------------
def text_to_embedding(text, vocab, word_embeddings):
    vectors = []
    for word in text.split():
        if word in vocab:
            vectors.append(word_embeddings[vocab[word]])
    
    if len(vectors) == 0:
        return np.zeros(embedding_dim)
    
    return np.mean(vectors, axis=0)

# Create embedding matrix
embeddings = np.array([text_to_embedding(t, vocab, word_embeddings) for t in texts])

# ------------------------------
# 5. Cosine Similarity (Vectorized)
# ------------------------------
def cosine_similarity_matrix(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / norm
    return X_normalized @ X_normalized.T

similarity_matrix = cosine_similarity_matrix(embeddings)

# ------------------------------
# 6. Query Search (Top-K)
# ------------------------------
def search(query, texts, vocab, word_embeddings, k=2):
    query_vec = text_to_embedding(query, vocab, word_embeddings)
    
    # Normalize
    query_vec = query_vec / np.linalg.norm(query_vec)
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    scores = emb_norm @ query_vec   # vectorized
    
    top_k_idx = np.argsort(scores)[-k:][::-1]
    
    return [(texts[i], scores[i]) for i in top_k_idx]

# ------------------------------
# 7. Run Example
# ------------------------------
print("Vocabulary:\n", vocab)
print("\nEmbeddings Shape:", embeddings.shape)

print("\nCosine Similarity Matrix:\n", similarity_matrix)

query = "machine learning"
results = search(query, texts, vocab, word_embeddings, k=2)

print("\nTop Matches for Query:", query)
for text, score in results:
    print(f"{text}  -->  {score:.4f}")