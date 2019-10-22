from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

if __name__ == '__main__':
    chatbot = ChatBot(name='house')

    # Create a new trainer for the chatbot
    trainer = ChatterBotCorpusTrainer(chatbot)

    # Train the chatbot based on the english corpus
    trainer.train('chatterbot.corpus.english')

    # Get a response to an input statement
    response = chatbot.get_response('Hello, how are you today?')
    print(response)
