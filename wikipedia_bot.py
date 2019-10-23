from random import choice
from time import time

from nltk import download
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from wikipedia import page as wiki_page
from wikipedia import random as wiki_random
from wikipedia.exceptions import DisambiguationError

from main import greeting
from main import respond

if __name__ == '__main__':
    download('punkt')  # first-time use only
    download('wordnet')  # first-time use only
    lemmer = WordNetLemmatizer()

    sentences = list()
    count = 0
    while count < 10:
        random_title = wiki_random()
        print('{}'.format(random_title))
        if not random_title.startswith('List'):
            try:
                page = wiki_page(title=random_title)
            except DisambiguationError as disambiguation_error:
                continue
            count += 1
            print('{}'.format(page.content))
            text = ' '.join([item for item in page.content.split('\n') if '==' not in item and len(item) > 1])
            print(text)
            new_sentences = sent_tokenize(text)
            if 'is' in new_sentences[0] or 'was' in new_sentences[0]:
                print(new_sentences[0])
            sentences = sentences + new_sentences
    print('sentence count: {}'.format(len(sentences)))

    flag = True
    print('Yes? ')
    while flag:
        user_response = input().lower()
        if user_response not in {'bye', 'quit'}:
            if user_response in {'thanks', 'thank you'}:
                flag = False
                print(choice(['welcome', 'np', 'yw']))
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
            print(': ' + choice(['later', 'see ya', 'cya', 'bye']))
