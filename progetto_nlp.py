# -*- coding: utf-8 -*-
"""Progetto NLP

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KXIFo1eBBtUqUODuX0HSKeIVyaUSxntz
"""



"""## Pre-processing"""

import pandas as pd
import re


#!pip install unidecode
import unidecode

#!pip install nltk
import nltk

from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

#!pip install pyspellchecker
#from spellchecker import SpellChecker

#!pip install contractions
#import contractions

#from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn import model_selection, naive_bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import svm
import pickle



import joblib


import numpy as np

df = pd.read_csv('dataset/train.csv')

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# this functions remove things like [1/2] or (3/3)
def remove_progressive(tweet):
  return re.sub(r"([\[|\(]\s*\d+\s*\/\s*\d+\s*[\]|\)])", "", tweet)

def remove_numbers(tweet):
  return ''.join([i for i in tweet if not i.isdigit()])

# this functions remove square brackets that appear removing numbers and progressive 
def remove_squarebrackets(tweet):
  return re.sub(r"\[\.*\w*\\/]", "", tweet)

def remove_nonunicode(tweet):
  return bytes(unidecode.unidecode(tweet), 'utf-8').decode('utf-8', 'ignore')

def remove_symbols(tweet):
  return re.sub("[!@$+%#*:<?=>\"|.,()'_-]", ' ', tweet)

def normalize_slangs(tweet):
  dictionary = {'u':'you', 'ppl':'people', 'bc':'because', 'bs': 'bullshit', 
                'w/o': 'without', 'luv':'love', 'prob': 'probably', 'wp': 'well played', 'gg': 'good game', 'yg': 'young gunner', 'fud':'food',
                'btw': 'by the way', 'omw': 'on my way', 'y\'all': 'you all', 'f**king': 'fucking', 'cud':'could','f***ing': 'fucking', 'kul':'cool',
                'a**hole': 'asshole', 'fyn':'fine', 'f***': 'fuck', 'yur': 'your', 'gr8': 'great', 'pdx': 'portland', 'govt': 'government',
                'yr': 'your', 'wud':'would','lyk':'like','wateva':'whatever','ttyl':'talk to you later', 'fam':'family', 'ty': 'thank you', 
                'omg':'oh my god', 'blvd': 'boulevard', 'bruh':'brother', 'hv': 'have', 'dy': 'day', 'bihday': 'birthday', 'impoant': 'important',
                'nutshel': 'nutshell', 'bihday': 'birthday', 'exactli': 'exactly', 'fishi': 'fishy', 'easili': 'easily'}
  return ' '.join([dictionary.get(word, word) for word in TweetTokenizer().tokenize(tweet)])

# correct lengthened words (e.g. woooooooooooow) 
def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

def get_tags(tweet):
  return ' '.join([word for word in TweetTokenizer().tokenize(tweet) if word.startswith('#')])

def remove_stopwords(tweet):
  return ' '.join([word for word in TweetTokenizer().tokenize(tweet) if word not in stopwords.words('english')])

def fix_contractions(tweet):
  return contractions.fix(tweet)

def spell_correction(tweet):
  spell = SpellChecker(distance=1)
  misspelled = spell.unknown(TweetTokenizer().tokenize(tweet))

  for word in misspelled:
    if not word.startswith("#"):
      tweet.replace(word, spell.correction(word))
  return tweet

def stemming(tweet):
  return ' '.join([ PorterStemmer().stem(word) for word in TweetTokenizer().tokenize(tweet)])

df['preprocessed_tweet'] = df['tweet'].apply(lambda tw : ' '.join([tweet for tweet in tw.split() if not tweet.startswith("@")]))
df['tags'] = df['preprocessed_tweet'].apply(lambda tw : get_tags(tw))
df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : remove_progressive(tw))
df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : normalize_slangs(tw))
df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : remove_numbers(tw))
df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : remove_squarebrackets(tw))
df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : remove_nonunicode(tw))
df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : reduce_lengthening(tw))
df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : remove_symbols(tw))
df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : tw.lower())
#df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : fix_contractions(tw))
#df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : spell_correction(tw))

df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : remove_stopwords(tw))
df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : stemming(tw))

df.head(10)

df.to_csv("trainset_preprocessed.csv")

"""## Classification

### Stratified sampling
"""

preprocessed_df = df.drop(columns=['id', 'tweet','tags'])

# split the dataset between train and test 

X_train, X_test, y_train, y_test = train_test_split(preprocessed_df.drop(['label'], axis = 1), list(preprocessed_df.label), test_size=0.3)

# the trainset is composed by 50% nohate tweet and 50% hate tweet

X_train['label'] = y_train                                      # add labels to train dataset

nohate = X_train[X_train['label'] == 0]                         # df with nohate
hate = X_train[X_train['label'] == 1]                           # df with hate

nohate_sample = nohate.sample(n = hate.shape[0])                # nohate sample of the dimension of the hate df

df_merged = pd.concat([hate, nohate_sample], axis = 0)          # merge df with nohate sample ad hate

"""### Classification"""

# Gaussian with tf-idf

vectorizer = TfidfVectorizer()

# train
X_tfid = vectorizer.fit_transform(df_merged["preprocessed_tweet"])

X_tfid = X_tfid.todense()

# test
X_test_tfid = vectorizer.transform(X_test["preprocessed_tweet"]).todense()
y_train = list(df_merged["label"])

#GNB = GaussianNB()

#GNB.fit(X_tfid, y_train)

#predict_gnb_tfidf = GNB.predict(X_test_tfid)
#accuracy_score(y_test, predict_gnb_tfidf)



# SVM with TF-IDF
svc_tfidf = svm.SVC()

fitted = svc_tfidf.fit(X_tfid, y_train)

# save fit to file for later use
joblib.dump(vectorizer, 'feature.joblib')
joblib.dump(fitted, 'svc_tfidf.joblib') 

#predict_svm_tfidf = svc_tfidf.predict(X_test_tfid)
#print(accuracy_score(y_test, predict_svm_tfidf))