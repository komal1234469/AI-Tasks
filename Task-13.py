import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download required data (run once)
nltk.download('stopwords')

# -----------------------------
# 1. Load Dataset
# -----------------------------
# Dataset format: label,text
# label: ham or spam

data = {
    "label": ["ham", "spam", "ham", "spam", "ham", "spam"],
    "text": [
        "Hey, how are you?",
        "Congratulations! You won a lottery. Claim now!",
        "Let's meet tomorrow",
        "Win money now!!! Click here",
        "Are you coming to class?",
        "Free entry in 1000 prize contest!!!"
    ]
}

df = pd.DataFrame(data)

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# -----------------------------
# 2. Preprocessing Function
# -----------------------------
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

# -----------------------------
# 4. Vectorization (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# 5. Model Training
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# -----------------------------
# 7. Prediction Function
# -----------------------------
def predict_spam(message):
    cleaned = preprocess_text(message)
    vectorized = vectorizer.transform([cleaned])
    result = model.predict(vectorized)[0]
    
    return "SPAM 🚨" if result == 1 else "HAM ✅"

# -----------------------------
# 8. Test Prediction
# -----------------------------
msg1 = "You won a free iPhone! Click now"
msg2 = "Hey, are we meeting today?"

print("\nMessage 1:", predict_spam(msg1))
print("Message 2:", predict_spam(msg2))