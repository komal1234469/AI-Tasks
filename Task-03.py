import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
reviews = [
    "This product is amazing and works perfectly",
    "I am very disappointed with the quality",
    "Delivery was fast and packaging was good",
    "Not worth the money, very bad experience"
]
for review in reviews:
    words = word_tokenize(review.lower())
    stemmed_words = []
    for w in words:
        stemmed_words.append(stemmer.stem(w))
    lemmatized_words = []
    for w in words:
        lemmatized_words.append(lemmatizer.lemmatize(w))
    print("Original Review: ", review)
    print("Tokenized Words: ", words)
    print("Stemmed Words: ", stemmed_words)
    print("Lemmatized Words: ", lemmatized_words)
   