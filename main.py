# adapted from
# https://medium.com/analytics-vidhya/building-a-simple-chatbot-in-python-using-nltk-7c8c8215ac6e
import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# WordNet is a semantically-oriented dictionary of English included in NLTK.
def get_lemma_tokens(arg_lemmer, tokens):
    return [arg_lemmer.lemmatize(token) for token in tokens]


def normalize(arg_text, arg_punctuation=dict((ord(punct), None) for punct in string.punctuation),
              arg_lemmer=nltk.stem.WordNetLemmatizer()):
    return get_lemma_tokens(arg_lemmer, nltk.word_tokenize(arg_text.lower().translate(arg_punctuation)))


def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response, arg_tokens):
    arg_tokens.append(user_response)
    vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english', )
    tfidf = vectorizer.fit_transform(arg_tokens)
    similarity = cosine_similarity(tfidf[-1], tfidf)
    index = similarity.argsort()[0][-2]
    flat = similarity.flatten()
    flat.sort()
    found = flat[-2]
    result = arg_tokens[index] if found else 'sorry please try again.'
    return result


GREETING_INPUTS = ('hello', 'hi', 'greetings', 'sup', 'what\'s up', 'hey',)
GREETING_RESPONSES = ['hi', 'hey', '*nods*', 'hi there', 'hello', 'hooray!']

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
        raw_text = input_fp.read()

    text = raw_text.lower()
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
                    print(response(user_response, sentences))
                    sentences.remove(user_response)
        else:
            flag = False
            print(': ' + random.choice(['later', 'see ya', 'cya', 'bye']))
