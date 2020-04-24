import pickle
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
import string
import numpy as np

model=pickle.load(open('email_classifer','rb'))
vectorizer=pickle.load(open('vectorizer','rb'))

l=[]
a=input("enter the message : ")
l.append(a)
a=vectorizer.transform(l)

predictions=model.predict(a)
print(predictions)
