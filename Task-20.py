import numpy as np

try:
    from scipy.spatial.distance import cosine as scipy_cosine
    from scipy.spatial.distance import euclidean as scipy_euclidean
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class EmbeddingSimilarity:
    def __init__(self, normalize=False):
        """
        normalize: if True, vectors will be normalized automatically
        """
        self.normalize = normalize

    def _to_numpy(self, vec):
        vec = np.array(vec, dtype=float)
        if vec.ndim != 1:
            raise ValueError("Input must be a 1D vector")
        return vec

    def _normalize(self, vec):
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def _prepare(self, vec1, vec2):
        v1 = self._to_numpy(vec1)
        v2 = self._to_numpy(vec2)

        if v1.shape != v2.shape:
            raise ValueError("Vectors must have same dimensions")

        if self.normalize:
            v1 = self._normalize(v1)
            v2 = self._normalize(v2)

        return v1, v2

    # ----------------------------
    # Similarity / Distance Methods
    # ----------------------------

    def cosine_similarity(self, vec1, vec2):
        v1, v2 = self._prepare(vec1, vec2)

        dot_product = np.dot(v1, v2)
        norm_a = np.linalg.norm(v1)
        norm_b = np.linalg.norm(v2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def euclidean_distance(self, vec1, vec2):
        v1, v2 = self._prepare(vec1, vec2)
        return np.linalg.norm(v1 - v2)

    def dot_product(self, vec1, vec2):
        v1, v2 = self._prepare(vec1, vec2)
        return np.dot(v1, v2)

    def manhattan_distance(self, vec1, vec2):
        v1, v2 = self._prepare(vec1, vec2)
        return np.sum(np.abs(v1 - v2))

    # ----------------------------
    # SciPy Versions (Optional)
    # ----------------------------

    def cosine_similarity_scipy(self, vec1, vec2):
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy not installed")
        return 1 - scipy_cosine(vec1, vec2)

    def euclidean_distance_scipy(self, vec1, vec2):
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy not installed")
        return scipy_euclidean(vec1, vec2)

    # ----------------------------
    # Batch Processing
    # ----------------------------

    def batch_cosine_similarity(self, query_vec, matrix):
        query_vec = self._to_numpy(query_vec)
        matrix = np.array(matrix, dtype=float)

        if matrix.ndim != 2:
            raise ValueError("Matrix must be 2D")

        if matrix.shape[1] != query_vec.shape[0]:
            raise ValueError("Dimension mismatch")

        if self.normalize:
            query_vec = self._normalize(query_vec)
            matrix = np.array([self._normalize(row) for row in matrix])

        dot_products = matrix @ query_vec
        query_norm = np.linalg.norm(query_vec)
        matrix_norms = np.linalg.norm(matrix, axis=1)

        return dot_products / (matrix_norms * query_norm)

    # ----------------------------
    # Top-K Similarity Search
    # ----------------------------

    def top_k_similar(self, query_vec, matrix, k=3):
        scores = self.batch_cosine_similarity(query_vec, matrix)
        top_k_indices = np.argsort(scores)[::-1][:k]
        return top_k_indices, scores[top_k_indices]


# ----------------------------
# Example Usage (Main)
# ----------------------------

if __name__ == "__main__":

    vec1 = [0.2, 0.8, 0.5]
    vec2 = [0.1, 0.9, 0.4]

    sim = EmbeddingSimilarity(normalize=False)

    print("---- Basic Similarities ----")
    print("Cosine Similarity:", sim.cosine_similarity(vec1, vec2))
    print("Euclidean Distance:", sim.euclidean_distance(vec1, vec2))
    print("Dot Product:", sim.dot_product(vec1, vec2))
    print("Manhattan Distance:", sim.manhattan_distance(vec1, vec2))

    if SCIPY_AVAILABLE:
        print("\n---- SciPy Versions ----")
        print("Cosine (SciPy):", sim.cosine_similarity_scipy(vec1, vec2))
        print("Euclidean (SciPy):", sim.euclidean_distance_scipy(vec1, vec2))

    # ----------------------------
    # Batch Example
    # ----------------------------

    print("\n---- Batch Similarity ----")

    query = [0.2, 0.8, 0.5]
    dataset = [
        [0.1, 0.9, 0.4],
        [0.9, 0.1, 0.3],
        [0.2, 0.75, 0.55],
        [0.0, 0.2, 0.9]
    ]

    scores = sim.batch_cosine_similarity(query, dataset)
    print("Similarity Scores:", scores)

    # ----------------------------
    # Top-K Example
    # ----------------------------

    print("\n---- Top-K Similar ----")

    indices, top_scores = sim.top_k_similar(query, dataset, k=2)
    print("Top indices:", indices)
    print("Top scores:", top_scores)