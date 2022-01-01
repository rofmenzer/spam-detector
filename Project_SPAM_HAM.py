# -*- coding: utf-8 -*-
"""
data set from 
https://www.kaggle.com/uciml/sms-spam-collection-dataset
"""

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sklearn.metrics as model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

nltk.download('punkt')

nltk.download('stopwords')

nltk.download('wordnet')


#read Dataset
dataset=pd.read_csv('spam.csv',encoding='latin-1')#names=["v1","v2"])
#disply My dataset
print(dataset)
#get Text  
sent=dataset.iloc[:,[1]]['v2']
#display spam & hams content 
print(sent)
#get labels 
label=dataset.iloc[:,[0]]['v1']
#display labels (spam/ham)
print(set(label))
#import encoder 
#turn spam /ham into numerical representation 
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
label=le.fit_transform(label)

print(label)

print(le.classes_)
#preprocessing spams/hams
# preprocessing spams/hams
import re

# https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
lemma = WordNetLemmatizer()
word_stemmer = PorterStemmer()


# POS_TAGGER_FUNCTION : TYPE 1

def pos_tagger(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:
    return None


sentences = []
for sen in sent:
  # remove non letter
  senti = re.sub('[^A-Za-z]', ' ', sen)
  # lower my spam/ham
  senti = senti.lower()
  # tokenize sentence->array
  words = word_tokenize(senti)
  # delete stop words
  word = [i for i in words if i not in stopwords.words('english')]
  # find the POS tag for each token (verb/adjective..)
  pos_tagged = nltk.pos_tag(word)
  # print(pos_tagged)
  # we use our own pos_tagger function to make things simpler to understand.
  wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

  # start lemmatize (base on tag's type)
  lemmatized_sentence = []
  for word, tag in wordnet_tagged:
    if tag is None:
      # if there is no available tag, append the token as is
      lemmatized_sentence.append(word)
    else:
      # else use the tag to lemmatize the token
      lemmatized_sentence.append(lemma.lemmatize(word, tag))
  lemmatized_sentence = " ".join(lemmatized_sentence)
  sentences.append(lemmatized_sentence)
# print sentence after pre proessing

print(sentences)
#change textuelle representation to numerical one using tfidf 
#https://medium.com/@cmukesh8688/tf-idf-vectorizer-scikit-learn-dbc0244a911a
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(max_features=5000)

features=tfidf.fit_transform(sentences)

features=features.toarray()
#display features,;
print(features)

print(len(tfidf.get_feature_names()))

print(tfidf.get_feature_names())
#split data to Train/test 
feature_train,feature_test,label_train,label_test=train_test_split(features,label,test_size=0.2,random_state=7)
print("#"*100)
print("""#Naive Bayies""")

model=MultinomialNB()
model.fit(feature_train,label_train)

label_pred=model.predict(feature_test)
#predicted labels ;
print(label_pred)
#real labels 
print(label_test)

print("accuracy :",m.accuracy_score(label_test,label_pred))
print('classification report')
print(m.classification_report(label_test,label_pred))
print( 'confusion matrix')
print(m.confusion_matrix(label_test,label_pred))
print('#'*100)
print("""#SVC""")

model=SVC(kernel='linear')
model.fit(feature_train,label_train)

label_pred=model.predict(feature_test)

print(m.accuracy_score(label_test,label_pred))

print(label_pred)

print(label_test)

print(m.classification_report(label_test,label_pred))

print(m.confusion_matrix(label_test,label_pred))
print('#'*100)
print("""#LogisticRegression""")

model=LogisticRegression()
model.fit(feature_train,label_train)

label_pred=model.predict(feature_test)

print(m.accuracy_score(label_test,label_pred))

print( label_pred)

print(label_test)

print(m.classification_report(label_test,label_pred))

print(m.confusion_matrix(label_test,label_pred))
print('#'*100)
print("""#Decision Tree""")

model=DecisionTreeClassifier()
model.fit(feature_train,label_train)

label_pred=model.predict(feature_test)

print(m.accuracy_score(label_test,label_pred))

print(label_pred)

print(label_test)

print(m.classification_report(label_test,label_pred))

print(m.confusion_matrix(label_test,label_pred))

