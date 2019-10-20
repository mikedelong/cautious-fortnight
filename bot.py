# adapted from
# https://chatbotslife.com/how-to-create-an-intelligent-chatbot-in-python-c655eb39d6b1

from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

small = ['hi there!', 'hi!', 'how do?', 'how you?', 'cool.', 'fine, you?', 'extra cool.', 'okay',
         'glad to hear it.', 'awesome', 'excellent', 'not so good', 'sorry to hear that.', 'what\'s your name?',
         'i\'m bot. ask me a math question, please.']
math0 = ['pythagorean theorem', 'a squared plus b squared equals c squared.']
math1 = ['law of cosines', 'c**2 = a**2 + b**2 - 2 * a * b * cos(gamma)']

if __name__ == '__main__':
    bot = ChatBot(name='bot', read_only=True,
                  logic_adapters=['chatterbot.logic.MathematicalEvaluation', 'chatterbot.logic.BestMatch'])

    list_trainer = ListTrainer(chatbot=bot)
    for item in (small, math0, math1):
        list_trainer.train(item)

    response = None
    while response != 'quit':
        response = input('?: ')
        if response != 'quit':
            print(bot.get_response(response))
