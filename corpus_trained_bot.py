from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

if __name__ == '__main__':
    chatbot = ChatBot(name='house')
    ChatterBotCorpusTrainer(chatbot).train('chatterbot.corpus.english')

    done = False
    while not done:
        user_input = input('?: ')
        if user_input not in {'quit', 'bye', 'cya'}:
            print(chatbot.get_response(user_input))
        else:
            done = True
