import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

df=pd.read_csv('email.csv')

vectorizer=TfidfVectorizer(stop_words=stopwords.words('english')+list(string.punctuation),lowercase=True,max_features=1500,min_df=5,max_df=0.7)
TfidfVectorizer
messages_bow=vectorizer.fit_transform(df['message'].values).toarray()

X_train,X_test,y_train,y_test=train_test_split(messages_bow,df['label'].values,test_size=0.20,random_state=0)
classifier=MultinomialNB()

classifier.fit(X_train,y_train)



pred=classifier.predict(X_train)
print(classification_report(y_train,pred))
print(confusion_matrix(y_train,pred))
print(accuracy_score(y_train,pred))


predtest=classifier.predict(X_test)
print("test values")
print(classification_report(y_test,predtest))
print(confusion_matrix(y_test,predtest))
print(accuracy_score(y_test,predtest))


examples=['Free now!!!','shigeru kiritanus','The','Hi Bob, how about a game of golf tomorrow?','thanku','Congratulations, you won a 1 lack']

example_counts=vectorizer.transform(examples)

predictions=classifier.predict(example_counts)
print(predictions)

pickle.dump(classifier,open('email_classifer','wb'))
pickle.dump(vectorizer,open('vectorizer','wb'))

