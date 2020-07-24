# Hate Detection on Tweets

This project includes the source code of a telegram bot, which takes a twitter username as input and returns the response "I think that the user is not using hate speech" if the user is not using hate speech in the last 50 tweets or "I think that the user is using hate speech. I'm percentage% sure.", where *percentage* is the percentage of hate tweets.

This version of the project classify the tweets with a SVM classifier ad tf-idf, using a preprocessed dataset. The original dataset can be downloaded [here](https://www.kaggle.com/vkrahul/twitter-hate-speech).
It is possible to see the scores of others classificators (SVM, GNB) with others feature extraction mathods (tf-idf, BoW, GloVe, w2v) in [this notebook](https://colab.research.google.com/drive/1KXIFo1eBBtUqUODuX0HSKeIVyaUSxntz?usp=sharing).

## Contents
The project is composed of five main files:
- bot.py: is the "main" of the project, where the bot commands and behaviours are defined. All the other functions declared in the other files are used here.
- classification.py: contains the function to run the classification with the SVM classifier. The prefitted model can be found in `/dataset/svc_tfidf.joblib` and the prefitted tf-idf can be found in `dataset/feature.joblib`
- preprocessing.py: contains the functions needed to preprocess and normalize the tweets, in order, this task are perfromed: removal of urls, removal of progressive (e.g tweets with [1/2]), removal square brackets, removal of non unicode character, removal of symbols, lowerization, reduction of lengthening (e.g. change woooow to wow), tokenization, removal of numbers, normalization of slangs, removal of stopwords and the stemming.
- progetto_nlp.py: script to create the prefitted model and the prefitted tf-idf vector
- tweet_apy.py: contains the function to get the tweets given an username

## How to run
To run the project you will need to install these modules: sklearn, python-telegram-bot, pandas, nltk, unidecode, numpy. It is also needed to create 5 enviroment variables, that have self-explanatory names:
1. TW_consumer_key
2. TW_consumer_secret
3. TW_access_key
4. TW_access_secret
5. BOT_TELEGRAM_KEY

To run the project simply run `python bot.py` and chat with your bot. To re-create the prefitted models and vector run `python progetto_nlp.py` and move the files to the `dataset` directory.
