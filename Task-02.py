import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
def tokenize_string(s):
    return word_tokenize(s.lower())
def sentence_tokenize_string(s):
    return sent_tokenize(s)
s=input()
print(tokenize_string(s))
print(sentence_tokenize_string(s))
 