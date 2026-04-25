import numpy as np

# -------------------------------
# 1. Sample embedding vectors
# -------------------------------
embeddings = np.array([
    [3.0, 4.0, 0.0, 0.0],
    [1.0, 2.0, 2.0, 1.0],
    [10.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0]   # edge case: zero vector
])

print("Original Embeddings:\n", embeddings)

# -------------------------------
# 2. L2 Normalization (numerical stability safe)
# -------------------------------
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

# avoid division by zero
norms = np.where(norms == 0, 1, norms)

normalized_embeddings = embeddings / norms

print("\nNormalized Embeddings:\n", normalized_embeddings)

# -------------------------------
# 3. Similarity comparison function
# -------------------------------
def cosine_similarity_matrix(X):
    """Compute cosine similarity matrix"""
    return np.dot(X, X.T)

def dot_product_matrix(X):
    """Compute raw dot product matrix"""
    return np.dot(X, X.T)

# -------------------------------
# 4. Compute similarities
# -------------------------------
raw_similarity = dot_product_matrix(embeddings)
norm_similarity = cosine_similarity_matrix(normalized_embeddings)

print("\nRaw Dot Product Similarity:\n", raw_similarity)
print("\nCosine Similarity (After Normalization):\n", norm_similarity)

# -------------------------------
# 5. Numerical stability check
# -------------------------------
print("\nVector norms before normalization:\n", np.linalg.norm(embeddings, axis=1))
print("\nVector norms after normalization:\n", np.linalg.norm(normalized_embeddings, axis=1))

# -------------------------------
# 6. Demonstration of stability improvement
# -------------------------------
a = np.array([10.0, 0.0])
b = np.array([1.0, 0.0])

print("\n--- Simple Example ---")

print("Dot product (unstable):", np.dot(a, b))

a_norm = a / np.linalg.norm(a)
b_norm = b / np.linalg.norm(b)

print("Cosine similarity (stable):", np.dot(a_norm, b_norm))