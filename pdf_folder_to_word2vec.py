from glob import glob
from json import load as json_load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from string import punctuation
from time import time

import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from tika import parser
from unidecode import unidecode

PUNCTUATION = set(punctuation)


def ispunct(arg):
    for character in arg:
        if character not in PUNCTUATION:
            return False
    return True


if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started.')

    with open('./pdf_folder_to_tfidf.json', 'r') as settings_fp:
        settings = json_load(settings_fp, cls=None, object_hook=None, parse_float=None, parse_int=None,
                             parse_constant=None, object_pairs_hook=None)

    filter_threshold = settings['filter_threshold'] if 'filter_threshold' in settings.keys() else 1
    if 'filter_threshold' in settings.keys():
        logger.info('filter threshold: {}'.format(filter_threshold))
    else:
        logger.warning('filter threshold not in settings; using default {}'.format(filter_threshold))
    input_folder = settings['input_folder'] if 'input_folder' in settings.keys() else None
    if input_folder:
        logger.info('input folder: {}'.format(input_folder))
    else:
        logger.warning('input folder is None. Quitting.')
        quit(code=1)
    output_file = settings['output_file'] if 'output_file' in settings.keys() else None
    if output_file:
        logger.info('output file: {}'.format(output_file))
    else:
        logger.warning('output file is missing from the settings. Quitting.')
        quit(code=2)
    plurals = settings['plurals'] if 'plurals' in settings.keys() else dict()
    if len(plurals):
        with open(settings['plurals'], 'r') as plurals_fp:
            plurals = json_load(plurals_fp)
            logger.info('plurals: {}'.format(plurals))
    else:
        logger.warning('plurals not in settings; we will not be doing any singular/plural reconciliation')
    stop_word = settings['stop_word'] if 'stop_word' in settings.keys() else list()
    if len(stop_word):
        with open(stop_word, 'r') as stop_word_fp:
            stop_words = json_load(stop_word_fp)
        if 'stop_word' in stop_words.keys():
            stop_word = stop_words['stop_word']
        else:
            logger.warning('stop word list malformed; check {}.'.format(settings['stop_word']))
            quit(code=4)
        logger.info('stop word list: {}'.format(stop_word))
    else:
        logger.warning('stop word list not in settings; default is empty.')

    verbs = settings['verbs'] if 'verbs' in settings.keys() else dict()
    if len(verbs):
        with open(settings['verbs'], 'r') as verbs_fp:
            verbs = json_load(verbs_fp)
            logger.info('verbs: {}'.format(verbs))
    else:
        logger.warning('verbs not in settings; we will not be doing any verbal-form consolidation')

    # do a sanity check on our plurals and verbs
    collisions = set(plurals.keys()).intersection(set(verbs.keys()))
    if len(collisions):
        logger.warning('we have plural/verb collisions: {}. Qutting.'.format(collisions))
        quit(code=3)

    input_files = [input_file for input_file in glob(input_folder + '*.pdf')]
    items = list()
    for input_file in input_files:
        parse_result = parser.from_file(input_file)
        if parse_result['content']:
            items.append(unidecode(parse_result['content']))
            logger.info('length: {} name: {}'.format(len(parse_result['content']), input_file))
        else:
            logger.warning('length: 0 name: {}'.format(input_file))

    logger.info('result size: {}'.format(len(items)))
    logger.info('file count: {}'.format(len(input_files)))

    # add a map of singulars to plurals to complement our plurals to singulars map
    singulars = {plurals[key]: key for key in plurals.keys()}

    # todo: factor these out as data
    capitalization = {'AFFAIRS', 'AID', 'AMBASSADOR', 'Also', 'Attendees', 'But', 'Code', 'Coordination', 'Corruption',
                      'File', 'For', 'INTERVIEW', 'Interview', 'Key', 'LEARNED', 'LESSONS', 'Learned', 'Lessons',
                      'Location', 'Meeting', 'No', 'Number', 'OF', 'On', 'Our', 'Page', 'People', 'Please', 'Prepared',
                      'Project', 'Purpose', 'RECORD', 'Record', 'Recorded', 'Recording', 'Research', 'Reviewed',
                      'SUBJECT', 'So', 'Thanks', 'The', 'These', 'They', 'This', 'Title', 'To', 'Topics', 'Untitled',
                      'With', 'Yes', 'He', 'In', 'There', 'FROM', 'TO', 'At', 'Not', }
    logger.info('capitalization tokens: {}'.format(sorted(list(capitalization))))
    split = {'AFGHAN': ['Afghan'], 'AFGHANISTAN': ['Afghanistan'], 'AMERICA': ['America'],
             'AMERICA1:1': ['America'], 'ofthe': ['of', 'the'], 'Date/Time': ['date', 'time'],
             '(Name,title': ['name', 'title'], 'Recordof': ['record', 'of'], 'U.S.': ['US'], 'wantto': ['want', 'to'],
             'wasa': ['was', 'a'], }

    text = list()
    for item_index, item in enumerate(items):
        if item is not None:
            pieces = [piece.strip() for piece in item.split()]
            pieces = [[piece] if piece not in split.keys() else split[piece] for piece in pieces]
            pieces = [item for piece in pieces for item in piece]
            pieces = [piece[1:] if piece.startswith('(') and ')(' not in piece else piece for piece in pieces]
            pieces = [piece[:-1] if piece.endswith(')') and ')(' not in piece else piece for piece in pieces]
            for punctuation in ['\'', '\"', '[', ]:
                pieces = [piece if not piece.startswith(punctuation) else piece[1:] for piece in pieces]
            for punctuation in [':', ';', '.', ',', '?', '\'', '\"', ']', '|', '/']:
                pieces = [piece if not piece.endswith(punctuation) else piece[:-1] for piece in pieces]
            pieces = [piece for piece in pieces if len(piece) > 1]
            pieces = [piece for piece in pieces if not piece.isdigit()]
            pieces = [piece if piece not in capitalization else piece.lower() for piece in pieces]
            pieces = [piece for piece in pieces if not ispunct(piece)]
            text.append(' '.join(pieces))

    corpus = [item.split() for item in text]
    model = Word2Vec(corpus, size=100, window=20, min_count=120, workers=4, batch_words=True, )

    labels = [word for word in model.wv.vocab]
    tokens = [model.wv[word] for word in model.wv.vocab]
    logger.info('tokens with capitals: {}'.format([item for item in labels if str(item) != str(item).lower()]))

    random_state = 1
    tsne_model = TSNE(angle=0.5, early_exaggeration=12.0, init='pca', learning_rate=100.0,
                      method='barnes_hut',
                      # method='exact',
                      metric='euclidean', min_grad_norm=1e-07, n_components=2, n_iter=2500, n_iter_without_progress=300,
                      perplexity=40.0, random_state=random_state, verbose=1, )
    tsne_values = tsne_model.fit_transform(tokens)

    xs = [value[0] for value in tsne_values]
    ys = [value[1] for value in tsne_values]

    plt.figure(figsize=(16, 16))
    for i in range(len(tsne_values)):
        plt.scatter(xs[i], ys[i])
        plt.annotate(labels[i], ha='right', textcoords='offset points', va='bottom', xy=(xs[i], ys[i]), xytext=(5, 2), )
    plt.tick_params(axis='both', bottom='off', labelbottom='off', labelleft='off', left='off', right='off', top='off',
                    which='both', )
    plt.show()
