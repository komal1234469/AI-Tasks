import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer


# Run only once if needed:
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


def build_nlp_pipeline(texts, use_tfidf=False):
    """
    NLP Pipeline:
    1. Lowercase
    2. Remove URLs
    3. Remove numbers
    4. Remove punctuation
    5. Tokenization
    6. Stopword removal
    7. Lemmatization
    8. Optional TF-IDF vectorization
    """

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    cleaned_texts = []

    for text in texts:
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenize
        tokens = word_tokenize(text)

        # Stopword removal + Lemmatization
        tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word not in stop_words and len(word) > 2
        ]

        cleaned_texts.append(" ".join(tokens))

    # TF-IDF option
    if use_tfidf:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(cleaned_texts)
        return X, vectorizer

    return cleaned_texts


# ---------------- EXAMPLE ----------------
docs = [
    "I love Machine Learning! Visit https://ml.com",
    "NLP pipelines are very powerful in AI systems."
]

# Cleaned text output
cleaned = build_nlp_pipeline(docs)
print("Cleaned Text:")
print(cleaned)

# TF-IDF output
X, vectorizer = build_nlp_pipeline(docs, use_tfidf=True)
print("\nTF-IDF Shape:", X.shape)