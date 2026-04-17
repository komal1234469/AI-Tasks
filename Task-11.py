from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample documents
doc1 = "I love machine learning and data science. It is very interesting and useful."
doc2 = "I love machine learning and data science because it is very interesting and useful."

# Step 1: Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform([doc1, doc2])

# Step 2: Compute cosine similarity
similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

print("Cosine Similarity:", similarity[0][0])