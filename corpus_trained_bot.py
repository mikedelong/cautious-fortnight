from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

if __name__ == '__main__':
    chatbot = ChatBot(name='house')

    ChatterBotCorpusTrainer(chatbot).train('chatterbot.corpus.english')

    # Get a response to an input statement
    for _ in range(10):
        print(chatbot.get_response('hello'))

    print(chatbot.get_response('How do you solve a problem like Maria?'))
