# 🚀 NLP Learning Journey: Text Preprocessing & Embedding Similarity

## 🌱 About Me

I am learning **Natural Language Processing (NLP)** daily and building small-to-advanced projects to strengthen my understanding of text processing, embeddings, and similarity computation.

This project is a combination of:

* NLP Text Preprocessing Pipeline
* Word Frequency Analysis & Visualization
* Embedding Similarity Calculator (Custom + SciPy)

---

# 📌 Project Overview

This project demonstrates how raw text can be transformed into meaningful numerical representations and analyzed using NLP techniques.

It includes:

* Cleaning and preprocessing raw text
* Word frequency analysis
* Visualization of most common words
* Vector similarity calculations using embeddings

---

# 🧠 Features

## 🔹 NLP Pipeline

* Lowercasing text
* Removing numbers
* Tokenization
* Stopword removal
* Lemmatization
* Word frequency analysis
* Visualization using Matplotlib

## 🔹 Embedding Similarity Module

* Cosine similarity
* Euclidean distance
* Dot product
* Manhattan distance
* Batch similarity computation
* Top-K similarity search
* Optional SciPy-based optimization

---

# ⚙️ Tech Stack

* Python 🐍
* NumPy
* NLTK
* Matplotlib
* SciPy (optional)

---

# 📂 Project Structure

```
NLP-Project/
│
├── preprocessing.py        # NLP cleaning pipeline
├── visualization.py        # Word frequency chart
├── similarity.py           # Embedding similarity class
├── main.py                 # Driver code
└── README.md
```

---

# 🧹 NLP Text Preprocessing Workflow

1. Input raw text from user
2. Convert to lowercase
3. Remove numbers & punctuation
4. Tokenize text
5. Remove stopwords
6. Apply lemmatization
7. Count word frequency using Counter
8. Plot results using Matplotlib

---

# 📊 Word Frequency Visualization

The most frequent words in the text are visualized using a bar chart to understand text patterns and importance of words.

---

# 📐 Embedding Similarity Class (Core Logic)

```python
import numpy as np

try:
    from scipy.spatial.distance import cosine as scipy_cosine
    from scipy.spatial.distance import euclidean as scipy_euclidean
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class EmbeddingSimilarity:
    def __init__(self, normalize=False):
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

    def cosine_similarity(self, vec1, vec2):
        v1, v2 = self._prepare(vec1, vec2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def euclidean_distance(self, vec1, vec2):
        v1, v2 = self._prepare(vec1, vec2)
        return np.linalg.norm(v1 - v2)

    def dot_product(self, vec1, vec2):
        v1, v2 = self._prepare(vec1, vec2)
        return np.dot(v1, v2)

    def manhattan_distance(self, vec1, vec2):
        v1, v2 = self._prepare(vec1, vec2)
        return np.sum(np.abs(v1 - v2))

    def batch_cosine_similarity(self, query_vec, matrix):
        query_vec = np.array(query_vec, dtype=float)
        matrix = np.array(matrix, dtype=float)

        dot_products = matrix @ query_vec
        query_norm = np.linalg.norm(query_vec)
        matrix_norms = np.linalg.norm(matrix, axis=1)

        return dot_products / (matrix_norms * query_norm)

    def top_k_similar(self, query_vec, matrix, k=3):
        scores = self.batch_cosine_similarity(query_vec, matrix)
        top_k_indices = np.argsort(scores)[::-1][:k]
        return top_k_indices, scores[top_k_indices]
```



# 📈 Learning Outcome

* Learned full NLP preprocessing pipeline
* Understood vector similarity concepts
* Built reusable similarity class
* Practiced real-world ML/NLP foundation skills

---

# 🚀 Future Improvements

* Add Word Embeddings (Word2Vec / GloVe)
* Add BERT embeddings
* Build search engine using similarity
* Deploy as web app (Flask/Streamlit)

---

⭐ If you like this project, feel free to star the repository!
