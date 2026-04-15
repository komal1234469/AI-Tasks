import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# डाउनलोड (first time only)
nltk.download('punkt')
nltk.download('stopwords')

# ------------------------------
# Sample Text Data
# ------------------------------
texts = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love coding"
]

# ------------------------------
# Preprocessing Function
# ------------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())              # lowercase + tokenize
    tokens = [w for w in tokens if w.isalnum()]       # remove punctuation
    tokens = [w for w in tokens if w not in stop_words]  # remove stopwords
    return " ".join(tokens)

# Apply preprocessing
processed_texts = [preprocess(t) for t in texts]

print("Processed Text:")
print(processed_texts)
print("\n")

# ------------------------------
# Bag of Words (BoW)
# ------------------------------
bow_vectorizer = CountVectorizer()
bow_matrix = bow_vectorizer.fit_transform(processed_texts)

print("BoW Feature Names:")
print(bow_vectorizer.get_feature_names_out())

print("\nBoW Matrix:")
print(bow_matrix.toarray())
print("\n")

# ------------------------------
# TF-IDF
# ------------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_texts)

print("TF-IDF Feature Names:")
print(tfidf_vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())