# word2vec/T-SNE visualization modified from https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
from glob import glob
from json import load as json_load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from lxml import html
from sklearn.manifold import TSNE


def process(arg):
    print(arg)
    if arg.startswith('</') or arg.startswith('<!'):
        return None
    this = html.fromstring(arg)
    attributes = this.attrib
    if 'alt' in attributes.keys():
        result = attributes['alt']
        return result if result != 'Navy Biography' else None

    return None if str(this.text) == 'Download Official Photo' or str(this.text).startswith('Updated') else this.text


if __name__ == '__main__':
    random_state = 1
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started.')

    with open('./parse_bios.json', 'r') as settings_fp:
        settings = json_load(settings_fp, cls=None, object_hook=None, parse_float=None, parse_int=None,
                             parse_constant=None, object_pairs_hook=None)

    file_pattern = settings['file_pattern'] if 'file_pattern' in settings.keys() else None
    if file_pattern:
        logger.info('file pattern: {}'.format(file_pattern))
    else:
        logger.warning('file pattern is None. Quitting.')
        quit(code=1)

    input_folder = settings['input_folder'] if 'input_folder' in settings.keys() else None
    if input_folder:
        logger.info('input folder: {}'.format(input_folder))
    else:
        logger.warning('input folder is None. Quitting.')
        quit(code=2)

    all_text = list()
    input_files = [input_file for input_file in glob(input_folder + file_pattern)]
    missing = 0
    found = 0
    for input_file in input_files:
        input_file = input_file.replace('\\', '/')
        with open(input_file, 'r') as input_fp:
            input_read = input_fp.read()
            all_text.append(input_read)
            soup = BeautifulSoup(input_read, 'html.parser')
            h1 = soup.find_all('h1')
            p = soup.find_all('p')
            logger.info('file: {} size: {} h1: {} p: {}'.format(input_file, len(soup.text), len(h1), len(p), ))
            name = h1[1].contents[0].strip()
            t = p[1].find('h3')
            if hasattr(t, 'contents') and len(t.contents):
                logger.info('{} : {}'.format(name, t.contents[0]))
                found += 1
            else:
                logger.info('{} : no office'.format(name, ))
                missing += 1

    # let's implement our bespoke approach here
    # first split on newline to get lines within each file
    all_text = [item.replace('\n', ' ').replace('\t', ' ').strip().split('>') for item in all_text]
    # next get the set of distinct tokens by flattening
    tokens = set([item for sublist in all_text for item in sublist])
    common = {item for item in tokens if all([item in piece for piece in all_text])}
    net = [[(item + '>').strip() for item in sublist if item not in common] for sublist in all_text]
    net = [[process(item) for item in sublist] for sublist in net]
    net = [[item for item in sublist if item is not None] for sublist in net]
    lengths = [len(item) for item in net]
    corpus = [' '.join([item.replace('U.S.', 'US').replace('.', ' ').replace(',', ' ').replace(')', '').replace('(',
                                                                                                                '').replace(
        ';', ' ') for item in sublist]).split() for sublist in net]
    # stop_word = {'a', 'the', }
    # stop_word = {'at', 'also', 'of', 'and', 'on', 'with', 'from', 'in', 'In', }
    # stop_word = {'is', 'has', 'was', }
    # stop_word = {'he', 'He', 'his', 'His', }
    stop_word = {'a', 'also', 'an', 'and', 'as', 'at', 'for', 'from', 'has', 'he', 'his', 'in', 'is', 'of',
                 'on', 'she', 'the', 'to', 'was', 'where', 'with', }
    logger.info('stop word: {}'.format(sorted(list(stop_word), key=lambda x: x.lower())))
    corpus = [[item for item in sublist if item.lower() not in stop_word and len(item) > 1] for sublist in corpus]

    model = Word2Vec(corpus, size=100, window=20, min_count=70, workers=4, batch_words=True, )

    labels = [word for word in model.wv.vocab]
    tokens = [model.wv[word] for word in model.wv.vocab]

    # tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=random_state)
    tsne_model = TSNE(n_components=2, perplexity=40.0, early_exaggeration=12.0, learning_rate=100.0, n_iter=2500,
                      n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='pca', verbose=1,
                      random_state=random_state,
                      # method='barnes_hut',
                      method='exact',
                      angle=0.5, )
    tsne_values = tsne_model.fit_transform(tokens)

    xs = [value[0] for value in tsne_values]
    ys = [value[1] for value in tsne_values]

    plt.figure(figsize=(16, 16))
    for i in range(len(tsne_values)):
        plt.scatter(xs[i], ys[i])
        plt.annotate(labels[i], ha='right', textcoords='offset points', va='bottom', xy=(xs[i], ys[i]), xytext=(5, 2), )
    plt.show()

    logger.info('found/missing: {}/{}'.format(found, missing))
    logger.info('input file count: {}'.format(len(input_files)))
