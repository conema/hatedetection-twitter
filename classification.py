from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import pickle
import joblib 


import numpy as np

def classification(tweets):
    preprocessed_df = tweets.drop(columns=['tweet','tags'])

    # load fitted tfidf
    vectorizer = joblib.load("dataset/feature.joblib")

    # load fitted model
    svc_tfidf  = joblib.load("dataset/svc_tfidf.joblib")


    # vectorization to TF-IDF
    X_test_tfid = vectorizer.transform(preprocessed_df["preprocessed_tweet"]).todense()


    # SVM with TF-IDF
    predict = svc_tfidf.predict(X_test_tfid)

    return predict