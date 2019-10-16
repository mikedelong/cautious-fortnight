import nltk
import numpy as np
import random
import string


# WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(arg_lemmer, tokens):
    return [arg_lemmer.lemmatize(token) for token in tokens]


def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


if __name__ == '__main__':

    lemmer = nltk.stem.WordNetLemmatizer()
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

    do_basic_loop = False
    if do_basic_loop:
        response = ''
        while response not in {'quit'}:
            response = input('Yes? ')
            print(response)

    input_file = './data/chatbot.txt'
    errors_ = 'ignore'
    with open(input_file, 'r', errors=errors_) as input_fp:
        raw_text = input_fp.read()

    text = raw_text.lower()  # converts to lowercase
    nltk.download('punkt')  # first-time use only
    nltk.download('wordnet')  # first-time use only
    sent_tokens = nltk.sent_tokenize(text)  # converts to list of sentences
    word_tokens = nltk.word_tokenize(text)  # converts to list of words
    print(len(word_tokens))
