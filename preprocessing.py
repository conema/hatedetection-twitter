import pandas as pd
import re

import unidecode

import nltk

from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


# cleaning before tokenization
def remove_urls(tweet):
    return re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)", "", tweet)

# this functions remove things like [1/2] or (3/3)
def remove_progressive(tweet):
  return re.sub(r"([\[|\(]\s*\d+\s*\/\s*\d+\s*[\]|\)])", "", tweet)

# this functions remove square brackets that appear removing numbers and progressive 
def remove_squarebrackets(tweet):
  return re.sub(r"\[\.*\w*\\/]", "", tweet)

def remove_nonunicode(tweet):
  return bytes(unidecode.unidecode(tweet), 'utf-8').decode('utf-8', 'ignore')

def remove_symbols(tweet):
  return re.sub("[!@$+%#*:<?=>\"|.,()'_-]", ' ', tweet)

def get_tags(tweet):
  return[word for word in TweetTokenizer().tokenize(tweet) if word.startswith('#')]

# correct lengthened words (e.g. woooooooooooow) 
def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)

# cleaning after tokenization
def normalize_slangs(tweetList):
  dictionary = {'u':'you', 'ppl':'people', 'bc':'because', 'bs': 'bullshit', 
                'w/o': 'without', 'luv':'love', 'prob': 'probably', 'wp': 'well played', 'gg': 'good game', 'yg': 'young gunner', 'fud':'food',
                'btw': 'by the way', 'omw': 'on my way', 'y\'all': 'you all', 'f**king': 'fucking', 'cud':'could','f***ing': 'fucking', 'kul':'cool',
                'a**hole': 'asshole', 'fyn':'fine', 'f***': 'fuck', 'yur': 'your', 'gr8': 'great', 'pdx': 'portland', 'govt': 'government',
                'yr': 'your', 'wud':'would','lyk':'like','wateva':'whatever','ttyl':'talk to you later', 'fam':'family', 'ty': 'thank you', 
                'omg':'oh my god', 'blvd': 'boulevard', 'bruh':'brother', 'hv': 'have', 'dy': 'day', 'bihday': 'birthday', 'impoant': 'important',
                'nutshel': 'nutshell', 'exactli': 'exactly', 'fishi': 'fishy', 'easili': 'easily', 'Ima': 'i am going to',
                'Y’all': 'you all', 'urd': 'agree'}
  return [dictionary.get(word, word) for word in tweetList]

def remove_stopwords(tweetList):
  return [word for word in tweetList if word not in stopwords.words('english')]

def remove_numbers(tweet):
  return [i for i in tweet if not i.isdigit()]

#def spell_correction(tweet):
#  spell = SpellChecker(distance=1)
#  misspelled = spell.unknown(TweetTokenizer().tokenize(tweet))

#  for word in misspelled:
#    if not word.startswith("#"):
#      tweet.replace(word, spell.correction(word))
#  return tweet

def stemming(tweetList):
  return [ PorterStemmer().stem(word) for word in tweetList]


def preprocess_tweets(tweets):
  df = pd.DataFrame(tweets, columns =['tweet'])

  df['preprocessed_tweet'] = df['tweet'].apply(lambda tw : ' '.join([tweet for tweet in tw.split() if not tweet.startswith("@")]))
  #df['tags'] = df['preprocessed_tweet'].apply(lambda tw : get_tags(tw))

  df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : remove_urls(tw))
  df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : remove_progressive(tw))

  df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : remove_squarebrackets(tw))
  df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : remove_nonunicode(tw))

  df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : remove_symbols(tw))
  df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : tw.lower())
  df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : reduce_lengthening(tw))

  # tokenization
  df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw: TweetTokenizer().tokenize(tw))

  #df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : spell_correction(tw))
  df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : remove_numbers(tw))
  df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : normalize_slangs(tw))
  df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : remove_stopwords(tw))
  df['preprocessed_tweet'] = df['preprocessed_tweet'].apply(lambda tw : stemming(tw))


  return df