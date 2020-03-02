from json import load as json_load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from math import log
from ntpath import basename
from string import punctuation
from time import time

import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from matplotlib.pyplot import cm
from plotly.graph_objects import Figure
from plotly.graph_objects import Layout
from plotly.graph_objects import Scatter
from plotly.offline import plot
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from tika import parser
from unidecode import unidecode

PUNCTUATION = set(punctuation)


def float_color_to_hex(arg_float, arg_colormap):
    color_value = tuple([int(255 * arg_colormap(arg_float)[index]) for index in range(3)])
    return '#{:02x}{:02x}{:02x}'.format(color_value[0], color_value[1], color_value[2])


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

    with open('./pdf_file_to_word2vec.json', 'r') as settings_fp:
        settings = json_load(settings_fp, cls=None, object_hook=None, parse_float=None, parse_int=None,
                             parse_constant=None, object_pairs_hook=None)

    capitalization = list()
    capitalization_file = settings['capitalization'] if 'capitalization' in settings.keys() else list()
    if len(capitalization_file):
        with open(capitalization_file, 'r') as capitalization_fp:
            capitalization_data = json_load(capitalization_fp)
        if 'data' in capitalization_data.keys():
            capitalization = capitalization_data['data']
        else:
            logger.warning('capitalization fix list malformed; check {}.'.format(settings['capitalization']))
            quit(code=5)
        logger.info('capitalization fix list: {}'.format(capitalization))
    else:
        logger.warning('capitalization fix list not in settings; default is empty.')
    capitalization = set(capitalization)

    input_file = settings['input_file'] if 'input_file' in settings.keys() else None
    if input_file:
        logger.info('input folder: {}'.format(input_file))
    else:
        logger.warning('input file is None. Quitting.')
        quit(code=1)
    colormap = settings['colormap'] if 'colormap' in settings.keys() else 'jet'
    if 'colormap' in settings.keys():
        logger.info('plotly colormap: {}'.format(colormap))
    else:
        logger.warning('colormap not in settings; using default {}'.format(colormap))

    items = list()
    parse_result = parser.from_file(input_file)
    if parse_result['content']:
        items.append(unidecode(parse_result['content']))
        logger.info('length: {} name: {}'.format(len(parse_result['content']), input_file))
    else:
        logger.warning('length: 0 name: {}'.format(input_file))

    logger.info('capitalization tokens: {}'.format(sorted(list(capitalization))))
    split = {'AFGHAN': ['Afghan'], 'AFGHANISTAN': ['Afghanistan'], 'AMERICA': ['America'], 'AMERICA1:1': ['America'],
             'ofthe': ['of', 'the'], 'Date/Time': ['date', 'time'], '(Name,title': ['name', 'title'],
             'Recordof': ['record', 'of'], 'U.S.': ['US'], 'wantto': ['want', 'to'], 'wasa': ['was', 'a'],
             'Interviewees:(Eitherlist': ['interviewees', 'either', 'list'], 'DoD': ['DOD'], 'Gen': ['General'],
             'MMDDYY': ['MM', 'DD', 'YY'], 'MM/DD/YY': ['MM', 'DD', 'YY'], 'SIGARAttendees': ['SIGAR', 'attendees'],
             'china': ['China'], 'cn': ['CN'], 'chinese': 'Chinese', 'China\'s': ['China'], 'Beijing\'s': ['Beijing'],
             'Taiwan\'s': ['Taiwan'], }

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
    model = Word2Vec(corpus,
                     batch_words=20,
                     # batch_words=True,
                     compute_loss=True,
                     min_count=24, size=30, sorted_vocab=1,
                     window=36, workers=4, )

    labels = [word for word in model.wv.vocab]
    tokens = [model.wv[word] for word in model.wv.vocab]
    logger.info('tokens with capitals: {}'.format(sorted([item for item in labels if str(item) != str(item).lower()])))

    random_state = 1
    tsne_model = TSNE(angle=0.5, early_exaggeration=12.0, init='pca', learning_rate=100.0,
                      # method='barnes_hut',
                      method='exact',
                      metric='euclidean', min_grad_norm=1e-07, n_components=2, n_iter=2500, n_iter_without_progress=300,
                      perplexity=8.0, random_state=random_state, verbose=1, )
    tsne_values = tsne_model.fit_transform(tokens)

    xs = [value[0] for value in tsne_values]
    ys = [value[1] for value in tsne_values]

    # todo move this to settings
    approach = 'plotly'
    if approach == 'matplotlib':
        plt.figure(figsize=(16, 16))
        for i in range(len(tsne_values)):
            plt.scatter(xs[i], ys[i])
            plt.annotate(labels[i], ha='right', textcoords='offset points', va='bottom', xy=(xs[i], ys[i]),
                         xytext=(5, 2), )
        plt.tick_params(axis='both', bottom=False, labelbottom=False, labelleft=False, left=False, right=False,
                        top=False, which='both', )
        plt.show()
    elif approach == 'plotly':
        vectorizer = CountVectorizer(lowercase=False)
        fit_result = vectorizer.fit_transform(text)
        result = dict(zip(vectorizer.get_feature_names(), fit_result.toarray().sum(axis=0)))
        result = {key: int(result[key]) for key in result.keys() if str(key) in labels}
        min_count = min(result.values())
        max_count = max(result.values())
        logger.info('count: min: {} max: {}'.format(min_count, max_count))
        misses = [item for item in labels if item not in result.keys()]
        logger.warning('label misses: {}'.format(misses))
        labels = [item for item in labels if item in result.keys()]
        colors = [
            float_color_to_hex(int(255 * log(1.0 + float(result[this] - min_count) / float(max_count - min_count))),
                               cm.get_cmap(colormap)) for this in labels]

        figure = Figure(Scatter(hoverinfo='text', hovertext=['{}: {}'.format(item, result[item], ) for item in labels],
                                mode='text', text=labels, textfont=dict(color=colors, ), x=xs, y=ys, ),
                        layout=Layout(autosize=True, xaxis=dict(showticklabels=False),
                                      yaxis=dict(showticklabels=False), ))
        output_file = './' + basename(input_file).replace('.pdf', '_word2vec.') + 'html'
        logger.info('saving HTML figure to {}'.format(output_file))
        plot(auto_open=False, auto_play=False, figure_or_data=figure, filename=output_file,
             link_text='', output_type='file', show_link=False, validate=True, )
    else:
        raise ValueError('plotting approach is {}. Quitting.'.format(approach))
