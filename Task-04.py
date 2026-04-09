import nltk
import string
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
reviews = [
    "This product is amazing!!! I loved it 100%",
    "Worst product ever... waste of money 123",
    "Delivery was fast, but packaging was bad!!!"
]
cleaned=[]
for i in reviews:
    words=word_tokenize(i.lower())
    for word in words:
        if word not in string.punctuation and word not in stop_words and word not in string.digits:
            cleaned.append(word)
print(cleaned)

