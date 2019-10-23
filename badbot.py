# adapted from
# https://chatbotslife.com/how-to-create-an-intelligent-chatbot-in-python-c655eb39d6b1

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

THIS = list()
THAT = list()

if __name__ == '__main__':
    bot = ChatBot(logic_adapters=['chatterbot.logic.BestMatch'], name='bot', read_only=True, )

    list_trainer = ListTrainer(chatbot=bot)
    for item in (THIS, THAT):
        list_trainer.train(item)

    done = False
    while not done:
        user_input = input('?: ')
        if user_input not in {'bye', 'cya', 'q', 'quit'}:
            print(bot.get_response(user_input))
        else:
            done = True
