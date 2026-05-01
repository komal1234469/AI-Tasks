# Task 23 - Feature Engineering Basics
# Convert Text into TF-IDF Features and Train a Classifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
texts = [
    "I love machine learning",
    "AI is the future",
    "Python is easy to learn",
    "I hate bugs in code",
    "Debugging is very frustrating",
    "Machine learning is amazing",
    "Coding is fun",
    "Errors make programming difficult"
]

# Labels
# 1 = Positive
# 0 = Negative
labels = [1, 1, 1, 0, 0, 1, 1, 0]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Test with custom sentence
sample_text = ["Machine learning with Python is fun"]

sample_tfidf = vectorizer.transform(sample_text)

prediction = model.predict(sample_tfidf)

if prediction[0] == 1:
    print("\nPrediction: Positive")
else:
    print("\nPrediction: Negative")