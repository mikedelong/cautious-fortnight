# adapted from
# https://chatbotslife.com/how-to-create-an-intelligent-chatbot-in-python-c655eb39d6b1

from chatterbot import ChatBot

bot = ChatBot(name='bot', read_only=True,
              logic_adapters=['chatterbot.logic.MathematicalEvaluation', 'chatterbot.logic.BestMatch'])
