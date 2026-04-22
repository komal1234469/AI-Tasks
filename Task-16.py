import numpy as np

# Example embedding vectors
vec1 = np.array([0.2, 0.5, 0.1, 0.7])
vec2 = np.array([0.3, 0.6, 0.2, 0.8])

# 1. Dot Product (similarity between two vectors)
dot_product = np.dot(vec1, vec2)
print("Dot Product:", dot_product)

# 2. Matrix of embeddings (multiple vectors)
embeddings_A = np.array([
    [0.2, 0.5, 0.1, 0.7],
    [0.9, 0.1, 0.3, 0.4]
])

embeddings_B = np.array([
    [0.3, 0.6, 0.2, 0.8],
    [0.5, 0.2, 0.9, 0.1]
])

# 3. Matrix Multiplication
matrix_product = np.matmul(embeddings_A, embeddings_B.T)
print("Matrix Multiplication:\n", matrix_product)

# 4. Cosine Similarity (common in embeddings)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cos_sim = cosine_similarity(vec1, vec2)
print("Cosine Similarity:", cos_sim)