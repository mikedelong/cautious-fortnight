# adapted from
# https://chatbotslife.com/how-to-create-an-intelligent-chatbot-in-python-c655eb39d6b1

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer

COSINES = ['law of cosines', 'c**2 = a**2 + b**2 - 2 * a * b * cos(gamma)']
SMALL = ['hi there!', 'hi!', 'how do?', 'how you?', 'cool.', 'fine, you?', 'extra cool.', 'okay',
         'glad to hear it.', 'awesome', 'excellent', 'not so good', 'sorry to hear that.', 'what\'s your name?',
         'i\'m bot. ask me a math question, please.']
PYTHAGOREAN = ['pythagorean theorem', 'a squared plus b squared equals c squared.']

if __name__ == '__main__':
    bot = ChatBot(name='bot', read_only=True,
                  logic_adapters=['chatterbot.logic.MathematicalEvaluation', 'chatterbot.logic.BestMatch'])

    list_trainer = ListTrainer(chatbot=bot)
    for item in (SMALL, PYTHAGOREAN, COSINES):
        list_trainer.train(item)

    corpus_trainer = ChatterBotCorpusTrainer(chatbot=bot)
    corpus_trainer.train('chatterbot.corpus.english')

    response = None
    while response != 'quit':
        response = input('?: ')
        if response != 'quit':
            print(bot.get_response(response))
