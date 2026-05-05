# Day 27: Mini Project - Sentiment Analysis Model (ML)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (you can replace with real dataset like IMDb / Twitter)
data = {
    "text": [
        "I love this product, it is amazing!",
        "This is the worst experience ever",
        "Absolutely fantastic service",
        "I hate it, very bad quality",
        "Really good and useful",
        "Not worth the money",
        "I am very happy with this",
        "Terrible, I will not buy again"
    ],
    "sentiment": [
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative"
    ]
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.25, random_state=42
)

# Build ML Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),          # Text → Numbers
    ("clf", LogisticRegression())          # Model
])

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Test with custom input
test_text = ["This product is really good and I love it"]
print("\nCustom Prediction:", model.predict(test_text))