__author__  = "Yuan Chen"
__version__ = "2020.06.16"
# install nltk first
import nltk.data
# install nltk all packages befor eunning
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
file1 = open("text1.txt", encoding='utf-8')
file2 = open('text3.txt', 'wb') 
data = file1.read()
file2.write('\n-----\n'.join(tokenizer.tokenize(data)).encode("utf-8")) 