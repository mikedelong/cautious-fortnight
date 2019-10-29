from nltk import download
from nltk.stem import WordNetLemmatizer
from wikipediaapi import Wikipedia

if __name__ == '__main__':
    for package in ['punkt', 'wordnet']:
        download(package)
    lemmer = WordNetLemmatizer()

    client = Wikipedia(language='en')
    name = 'Chatbot'
    page = client.page(name)
    if page.exists():
        print('page [{}] exists'.format(name))
    else:
        print('page [{}] does not exist'.format(name))
        quit(1)
