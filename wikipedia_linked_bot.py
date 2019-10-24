from random import choice

import wikipedia
from nltk import download
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from wikipedia.exceptions import PageError

from main import greeting
from main import respond


def get_sentences(page_name):
    try:
        local_page = wikipedia.page(title=page_name)
    except PageError as page_error:
        return list()
    local_text = ' '.join([item for item in local_page.content.split('\n') if '==' not in item and len(item) > 1])
    return sent_tokenize(local_text)


if __name__ == '__main__':
    download('punkt')  # first-time use only
    download('wordnet')  # first-time use only
    lemmer = WordNetLemmatizer()

    found_root = False
    random_title = None
    while not found_root:
        random_title = wikipedia.random()
        print('{}'.format(random_title))
        found_root = not random_title.startswith('List')

    sentences = get_sentences(random_title)
    new_sentences = get_sentences(random_title)
    for link in wikipedia.page(title=random_title).links:
        print(link)
        if not link.startswith('List'):
            sentences = sentences + get_sentences(link)

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
                    response = respond(user_response, sentences)
                    print(response)
                    sentences.remove(user_response)
        else:
            flag = False
            print(': ' + choice(['later', 'see ya', 'cya', 'bye']))
