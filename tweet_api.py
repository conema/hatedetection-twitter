import twitter, os

def get_tweets(user, count):
    api = twitter.Api(consumer_key=os.environ['TW_consumer_key'],
                consumer_secret=os.environ['TW_consumer_secret'],
                access_token_key=os.environ['TW_access_key'],
                access_token_secret=os.environ['TW_access_secret'],
                tweet_mode='extended')

    try:
        statuses = api.GetUserTimeline(screen_name=user, count=count, include_rts=False) #exclude_replies=True,
    except twitter.TwitterError:
        return None

    return [s.full_text for s in statuses]
