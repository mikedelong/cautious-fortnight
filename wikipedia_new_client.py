from nltk import download
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from wikipediaapi import Wikipedia

UNWANTED_HEADINGS = {'External links', 'References', 'See also', }


def fix_period_splice(arg):
    for index, character in enumerate(arg):
        if character == '.' and 0 < index < len(arg) - 1:
            if arg[index - 1].islower() and arg[index + 1].isupper():
                result = ''.join([arg[:index], '. ', arg[index:]])
                return fix_period_splice(result)
    return arg


def get_sentences(arg_page):
    content = ''.join([section.text for section in arg_page.sections if section.title not in UNWANTED_HEADINGS])
    local_text = fix_period_splice(content)
    return sent_tokenize(local_text)


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
    print(page.links.keys())
    try:
        sentences = ''.join([' '.join(get_sentences(item)) for item in
                             [page] + [client.page(link) for link in page.links.keys() if
                                       not any([link.startswith('List of'),
                                                link.startswith('Template:'),
                                                link.startswith('Template talk:'),
                                                link.startswith('Portal:')])]])
        print(sentences)
    except TypeError as type_error:
        print(type_error.args)
