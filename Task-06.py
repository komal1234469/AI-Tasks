import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
text = """
Natural Language Processing (NLP) is a field of artificial intelligence.
It involves analyzing and processing human language data.
NLP techniques are widely used in machine learning, deep learning, and AI.
"""
text = text.lower()  
text = re.sub(r'[^a-z\s]', '', text)
words = text.split()
word_counts = Counter(words)
words_array = np.array(list(word_counts.keys()))
counts_array = np.array(list(word_counts.values()))
sorted_indices = np.argsort(counts_array)[::-1]
top_n = 20
top_words = words_array[sorted_indices][:top_n]
top_counts = counts_array[sorted_indices][:top_n]
plt.figure()
plt.bar(top_words, top_counts)
plt.xticks(rotation=45)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Top 20 Word Frequency Distribution")
plt.tight_layout()
plt.show()