import logging

from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove)
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters,
                          ConversationHandler)

import os
import math

import pandas as pd
import tweet_api as ta
import preprocessing as pp
import classification as clf
from classification import same_x

token_api = os.environ['BOT_TELEGRAM_KEY']

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

USERNAME, CANCEL = range(2)


def start(update, context):
    update.message.reply_text(
        'Hi! Send me a twitter username to analyze, without the "@"\n'
        'Send /cancel to stop talking to me.\n\n',
        reply_markup=ReplyKeyboardRemove())

    return USERNAME

def analyze_user(user, n_tweet):
    tweets = ta.get_tweets(user, n_tweet)

    if (tweets is None):
        return None

    preprocessed_tweets = pp.preprocess_tweets(tweets)

    predicted = clf.classification(preprocessed_tweets)

    return predicted


def username(update, context):
    n_tweet = 50

    logger.info("Analyzing %s", update.message.text)
    update.message.reply_text('⏱ I \'m analyzing the user ⏱')

    predicted = analyze_user(update.message.text, n_tweet)

    if (predicted is None):
        update.message.reply_text('I cannot found this username.\nGive me another one or type /cancel')
        return USERNAME

    logger.info("Predict: " + str(sum(predicted))+ "   "+ str(len(predicted)))


    if (sum(predicted) <= (len(predicted)*0.2)):
        # if the user has <=20% hate tweets
        update.message.reply_text('I think that the user is not using hate speech')
    else:
        if (n_tweet > len(predicted)):
            n_tweet = len(predicted)

        update.message.reply_text('I think that the user is using hate speech. I\'m ' + str(math.floor(sum(predicted)*100/n_tweet)) + '% sure.')

    update.message.reply_text('Give me another username or type /cancel',
        reply_markup=ReplyKeyboardRemove())

    return USERNAME

def cancel(update, context):
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text('Bye! I hope we can talk again some day.',
                              reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END


def main():
    updater = Updater(token_api, use_context=True)
    dp = updater.dispatcher

    # Conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={
            USERNAME: [
                        MessageHandler(Filters.regex('^\/cancel'), cancel),
                        MessageHandler(Filters.regex('^[^@]'), username)
                    ]
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )

    dp.add_handler(conv_handler)

    # Start the Bot
    updater.start_polling()

    # Run the bot until Ctrl-C is pressed
    updater.idle()


if __name__ == '__main__':
    main()
