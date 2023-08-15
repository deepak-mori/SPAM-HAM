# Installing required libararies and importing them
#!pip install nltk
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import os
import string
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# Reading the dataset file
data = pd.read_csv('spam1.csv')

encode = LabelEncoder()
data['type'] = encode.fit_transform(data['type'])

# Checking for duplicate values 
data.duplicated().sum()

# Removing duplicate values 
data = data.drop_duplicates(keep='first')

# calculating number of characters in each mail
data['num_characters'] = data['text'].apply(len)

# num of words
data['num_words'] = data['text'].apply(lambda x:len(nltk.word_tokenize(x)))

# num of sentences
data['num_sentences'] = data['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

# Transforming text in lower case, removing special characters, removing stop words and punctuations
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

# Building the model
ps = PorterStemmer()
data['transformed_text'] = data['text'].apply(transform_text)
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(data['transformed_text']).toarray()
y = data['type'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
svc.fit(X_train, y_train)

# path stores the current directory
path = os.getcwd()
path1 = path+"\\test"

dir = path1
os.chdir(dir)

# Reading and Checking for the mails in test folder that is spam(1) or ham(0)
for i in os.listdir(dir):
    if i.endswith(".txt"):
        
        with open(i, 'r') as file:
            st = file.read()
            
            string_transform = transform_text(st)
            string_tfift = tfidf.transform([string_transform]).toarray()
            string_result = svc.predict(string_tfift)
            print(string_result)
            
os.chdir(path)