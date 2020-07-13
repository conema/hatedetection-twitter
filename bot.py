import logging

from telegram import (ReplyKeyboardMarkup, ReplyKeyboardRemove)
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters,
                          ConversationHandler)

import os
token_api = os.environ['BOT_TELEGRAM_KEY']

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

USERNAME, RESULT = range(2)


def start(update, context):
    update.message.reply_text(
        'Hi! Send me a twitter username to analyze, with this format: @username"\n'
        'Send /cancel to stop talking to me.\n\n',
        reply_markup=ReplyKeyboardRemove())

    return USERNAME


def username(update, context):
    logger.info("Analyzing %s", update.message.text)
    update.message.reply_text('⏱ I \'m analyzing the user ⏱')


    update.message.reply_text('I think that the user is using hate speech')

    return ConversationHandler.END


def cancel(update, context):
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text('Bye! I hope we can talk again some day.',
                              reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END


def main():
    # Create the Updater and pass it your bot's token
    updater = Updater(token_api, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # Add conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={
            USERNAME: [MessageHandler(Filters.regex('^@'), username)]
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )

    dp.add_handler(conv_handler)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C
    updater.idle()


if __name__ == '__main__':
    main()
