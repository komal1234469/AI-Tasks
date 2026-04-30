# Step 1: Import libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Sample dataset
texts = [
    "I love this product",
    "This is the worst thing ever",
    "Absolutely fantastic experience",
    "Not good at all",
    "Very happy with the service",
    "Terrible and disappointing"
]

labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Step 4: Convert text to numerical features
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 6: Predictions
y_pred = model.predict(X_test_vec)

# Step 7: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))