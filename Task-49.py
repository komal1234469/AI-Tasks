# Day 49: Build Spam Classifier (Logic Level)

# Install required libraries
# pip install scikit-learn pandas

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Step 1: Sample Dataset
# -----------------------------

messages = [

    "Congratulations! You won a free iPhone",
    "Claim your lottery prize now",
    "Limited time offer click here",
    "Win cash instantly",

    "Let's meet tomorrow",
    "Can you send the report?",
    "Project meeting at 5 PM",
    "Lunch plan for today"
]

labels = [
    "spam",
    "spam",
    "spam",
    "spam",

    "ham",
    "ham",
    "ham",
    "ham"
]

# -----------------------------
# Step 2: Convert Text to Features
# -----------------------------

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(messages)

y = labels

# -----------------------------
# Step 3: Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# -----------------------------
# Step 4: Train Classifier
# -----------------------------

model = MultinomialNB()

model.fit(X_train, y_train)

# -----------------------------
# Step 5: Evaluate Model
# -----------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n📊 Model Accuracy:")
print(round(accuracy * 100, 2), "%")

# -----------------------------
# Step 6: Test Custom Message
# -----------------------------

test_message = [
    "Congratulations! Claim your reward now"
]

test_vector = vectorizer.transform(
    test_message
)

prediction = model.predict(
    test_vector
)

print("\n📩 Test Message:")
print(test_message[0])

print("\n🤖 Prediction:")
print(prediction[0])

# -----------------------------
# Step 7: Key Learning
# -----------------------------

print("\n💡 Key Learning:")
print("Spam classifiers use text patterns and probabilities to distinguish spam messages from normal messages.")