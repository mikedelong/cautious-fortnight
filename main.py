# adapted from
# https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e
import random
import string
from time import time

import nltk
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# WordNet is a semantically-oriented dictionary of English included in NLTK.
def get_lemma_tokens(arg_lemmer, tokens):
    return [arg_lemmer.lemmatize(token) for token in tokens]


def normalize(arg_text, arg_punctuation, arg_lemmer=nltk.stem.WordNetLemmatizer()):
    return get_lemma_tokens(arg_lemmer, nltk.word_tokenize(arg_text.lower().translate(arg_punctuation)))


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


GREETING_INPUTS = ('hello', 'hi', 'greetings', 'sup', 'what\'s up', 'hey',)
GREETING_RESPONSES = ['hi', 'hey', '*nods*', 'hi there', 'hello', 'hooray!']
INDEX_CHOICE = {'highest': 0, 'probabilistically': 1}['highest']
STOP_WORDS = {word for word in {'ha', 'le', 'u', 'wa'}.union(ENGLISH_STOP_WORDS)}


def respond(arg_user_response, arg_tokens):
    arg_tokens.append(arg_user_response)
    vectorizer = TfidfVectorizer(lowercase=False, tokenizer=lambda x: normalize(x, dict(
        (ord(punct), None) for punct in string.punctuation), nltk.stem.WordNetLemmatizer()), stop_words=STOP_WORDS, )
    tfidf = vectorizer.fit_transform(arg_tokens)
    similarity = cosine_similarity(tfidf[-1], tfidf)
    if np.count_nonzero(similarity.flatten()) == 1:
        return 'sorry please try again.'
    else:
        index = similarity.argsort()[0][-2] if INDEX_CHOICE == 0 else \
            np.random.choice(range(len(similarity[0]) - 1), 1, p=(similarity[0][:-1] / np.sum(similarity[0][:-1])))[0]
        return arg_tokens[index]


if __name__ == '__main__':
    lemmer = nltk.stem.WordNetLemmatizer()
    do_basic_loop = False
    if do_basic_loop:
        response = None
        while response not in {'quit'}:
            response = input('Yes? ')
            print(response)

    input_file = './data/chatbot.txt'
    errors_ = 'ignore'
    with open(input_file, 'r', errors=errors_) as input_fp:
        text = input_fp.read()

    nltk.download('punkt')  # first-time use only
    nltk.download('wordnet')  # first-time use only
    sentences = nltk.sent_tokenize(text)  # converts to list of sentences
    words = nltk.word_tokenize(text)  # converts to list of words
    print('we have {} words'.format(len(words)))

    flag = True
    print('Yes? ')
    while flag:
        user_response = input().lower()
        if user_response not in {'bye', 'quit'}:
            if user_response in {'thanks', 'thank you'}:
                flag = False
                print(random.choice(['welcome', 'np', 'yw']))
            else:
                if greeting(user_response):
                    print(': ' + greeting(user_response))
                else:
                    print(': ', end='')
                    time_start = time()
                    response = respond(user_response, sentences)
                    print('{:5.2f}s : {} '.format(time() - time_start, response))
                    sentences.remove(user_response)
        else:
            flag = False
            print(': ' + random.choice(['later', 'see ya', 'cya', 'bye']))
