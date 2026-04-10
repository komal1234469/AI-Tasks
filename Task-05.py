import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
news=input() 
news=re.sub(r'\d+','',news)
words=word_tokenize(news.lower())
new=""
print("Original News: ", news)
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()
filtered_words = []
for i in words:
    if i not in string.punctuation and i not in stop_words :
        filtered_words.append(i)
print("After Cleaning:", " ".join(filtered_words))
cleaned_words=[]
for i in filtered_words:
    cleaned_words.append(lemmatizer.lemmatize(i))
print("Cleaned Words:", " ".join(cleaned_words))