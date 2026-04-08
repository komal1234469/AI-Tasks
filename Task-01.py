def tokenize_string(s):
    return s.lower().split()
def remove_stop_words(tokens):
    stop_words = ['the', 'is', 'in', 'and', 'to', 'of']
    return [i for i in tokens if i not in stop_words]
def count_words(tokens):
    freq={}
    for i in tokens:
        if i in freq:
            freq[i]+=1
        else:
            freq[i]=1
    return freq
   
s=input()
tokens = tokenize_string(s)
print("Tokens:", tokens)
filtered_tokens = remove_stop_words(tokens)
print("Filtered Tokens:", filtered_tokens)
word_count = count_words(filtered_tokens)
print("Word Count:", word_count)

