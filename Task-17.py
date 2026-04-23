from collections import Counter
from scipy.stats import multinomial, poisson, norm

# Input text
text = "apple banana apple orange banana apple mango"

# Step 1: Count words
word_counts = Counter(text.split())

# Step 2: Total words
total_words = sum(word_counts.values())

# Step 3: Probability distribution
prob_dist = {word: count / total_words for word, count in word_counts.items()}

# Step 4: Multinomial probability
counts = list(word_counts.values())
pvals = list(prob_dist.values())
multi_prob = multinomial.pmf(counts, n=total_words, p=pvals)

# Step 5: Poisson (example: 'apple')
lambda_ = prob_dist['apple'] * total_words
poisson_prob = poisson.pmf(word_counts['apple'], mu=lambda_)

# Step 6: Normal approximation
mean = lambda_
std = lambda_ ** 0.5
normal_prob = norm.pdf(word_counts['apple'], loc=mean, scale=std)

# Output
print("Word Counts:", word_counts)
print("Probability Distribution:", prob_dist)
print("Multinomial Probability:", multi_prob)
print("Poisson Probability (apple):", poisson_prob)
print("Normal Approximation (apple):", normal_prob)