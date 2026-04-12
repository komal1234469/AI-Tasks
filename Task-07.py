# 1. Imports
import nltk
import re
import string
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# Download 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
# 2. Input
news = input("Enter text: ")

print("\nOriginal News:", news)
# 3. Cleaning
news = re.sub(r'\d+', '', news)  # remove numbers
words = word_tokenize(news.lower())

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Remove stopwords & punctuation
filtered_words = []
for word in words:
    if word not in string.punctuation and word not in stop_words:
        filtered_words.append(word)

print("\nAfter Cleaning:", " ".join(filtered_words))
# 4. Lemmatization
cleaned_words = []
for word in filtered_words:
    cleaned_words.append(lemmatizer.lemmatize(word))

print("\nCleaned Words:", " ".join(cleaned_words))
# 5. Word Frequency (FIXED)
word_counts = Counter(cleaned_words)

words_array = np.array(list(word_counts.keys()))
counts_array = np.array(list(word_counts.values()))

# Sort in descending order
sorted_indices = np.argsort(counts_array)[::-1]

top_n = 10
top_words = words_array[sorted_indices][:top_n]
top_counts = counts_array[sorted_indices][:top_n]

# 6. Visualization
plt.figure()
plt.bar(top_words, top_counts)
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top Word Frequency")
plt.tight_layout()
plt.show()