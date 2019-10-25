from random import choice

import wikipedia
from nltk import download
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from wikipedia.exceptions import DisambiguationError
from wikipedia.exceptions import PageError

from main import greeting
from main import respond


def fix_period_splice(arg):
    for index, character in enumerate(arg):
        if character == '.' and 0 < index < len(arg) - 1:
            if arg[index - 1].islower() and arg[index + 1].isupper():
                result = ''.join([arg[:index], '. ', arg[index:]])
                return fix_period_splice(result)
    return arg


def get_sentences(page_name):
    try:
        local_page = wikipedia.page(title=page_name)
    except PageError as page_error:
        return list()
    # todo find a way to recover by picking one of the available choices
    except DisambiguationError as disambiguation_error:
        return list()
    # todo split smart and remove the References section
    content = local_page.content
    lines = [line for line in content.split('\n') if len(line) > 0]
    headings = [index for index, line in enumerate(lines) if line.startswith('==') or index == 0]
    headings.append(len(lines))
    t = list()
    for index, heading in enumerate(headings[:-1]):
        section = lines[heading:headings[index + 1]]
        if lines[heading] not in {'== External links ==', '== See also ==', '== References =='}:
            for line in section:
                if not line.startswith('=='):
                    t.append(line)
    local_text = fix_period_splice(' '.join([item for item in t]))
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
