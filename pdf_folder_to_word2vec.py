from glob import glob
from json import load as json_load
from logging import INFO
from logging import basicConfig
from logging import getLogger
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

    with open('./pdf_folder_to_word2vec.json', 'r') as settings_fp:
        settings = json_load(settings_fp, cls=None, object_hook=None, parse_float=None, parse_int=None,
                             parse_constant=None, object_pairs_hook=None)

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

    input_files = [input_file for input_file in glob(input_folder + '*.pdf')]
    items = list()
    for input_file in input_files:
        parse_result = parser.from_file(input_file)
        if parse_result['content']:
            items.append(unidecode(parse_result['content']))
            logger.info('length: {} name: {}'.format(len(parse_result['content']), input_file))
        else:
            logger.warning('length: 0 name: {}'.format(input_file))

    # todo: factor these out as data
    capitalization = {'ACTION', 'AID', 'AND', 'ARE', 'According', 'Accordingly', 'Acting', 'Action',
                      'Administration', 'Advisory', 'Affairs', 'After', 'Agency', 'Agreement', 'Agreements',
                      'All', 'Also', 'Although', 'An', 'Analysis', 'And', 'Any', 'MEMORANDUM', 'SECRET',
                      'AN', 'AS', 'AT', 'BE', 'BUT', 'BY', 'DATE', 'DOCUMENT', 'EVENT', 'FOR', 'FROM', 'GOVERNMENT',
                      'HAD', 'HAVE', 'HE', 'HIS', 'IF', 'II', 'III', 'IN', 'INFORMATION', 'IS', 'IT', 'IV', 'MAP',
                      'MILITARY', 'NATIONAL', 'NO', 'NOT', 'OF', 'ON', 'ONLY', 'OR', 'PRESIDENT', 'ROLLING', 'SEA',
                      'SECURITY', 'SENSITIVE', 'SHOULD', 'SOUTH', 'STATE', 'THAT', 'THE', 'THEY', 'THIS', 'THUNDER',
                      'TO', 'TOP', 'WAS', 'WE', 'WERE', 'WHICH', 'WITH', 'WOULD', 'Finally', 'During',
                      'Sent', 'Special', 'Information', 'Now', 'Not', 'Yet', 'Two', 'Vols', 'Tab', 'This',
                      'Do', 'Even', 'Be', 'In', 'Some', 'Study', 'Of', 'Therefore', 'The', 'They', 'Since',
                      'Working', 'You', 'Both', 'What', 'Such', }
    logger.info('capitalization tokens: {}'.format(sorted(list(capitalization))))
    proper = {'RUSK', 'GENERAL', 'INDOCHINA', 'SECRETARY', 'SAIGON', 'DEFENSE', 'ARMY', 'FORCES', 'VIETNAM', 'FRENCH',
              'VIETNAMESE', }
    logger.info('proper noun fixes: {}'.format(sorted(list(proper))))
    split = {'U.S.': ['US'], 'U.S': ['US'], 'Minh\'s': ['Minh'], 'President\'s': ['President'], 'Ho\'s': ['Ho'],
             'Amembassy': ['American', 'Embassy'], 'Apr': ['April'], 'Jan': ['January'], 'Feb': ['February'],
             'Mar': ['March'], 'Jun': ['June'], 'Jul': ['July'], 'Aug': ['August'], 'Sep': ['September'],
             'Oct': ['October'], 'Nov': ['November'], 'Dec': ['December'], 'budgetconsultation':
                 ['budget', 'consultation'], 'Msg': ['message'], 'msg': ['message'], 'Diem\'s': ['Diem'],
             'Viet-Nam': ['Vietnam'], }
    join = {'bomb-', 'bomb', 'bind-', 'group-', 'command-', 'liv-', 'follow-', 'boil-', 'belong-',
            'belong-', 'hear-', 'issu-', 'arm-', 'follow-', 'contemplat-', 'accord-', 'imprison-',
            'attack-'}
    joined = {'bombing', 'binding', 'grouping', 'commanding', 'living', 'following', 'boiling', 'belonging',
              'hearing', 'issuing', 'arming', 'following', 'contemplating', 'according', 'imprisoning',
              'attacking', }
    # todo add a token consolidation loop

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
            pieces = [piece if piece not in proper else piece.capitalize() for piece in pieces]
            pieces = [piece for piece in pieces if not ispunct(piece)]
            pieces = [(piece + pieces[index + 1]).replace('-', '') if piece in join else piece for index, piece in
                      enumerate(pieces)]
            clean = list()
            for index, piece in enumerate(pieces):
                if piece in {'ing', '-ing'}:
                    if pieces[index - 1] not in joined:
                        clean.append(piece)
                else:
                    clean.append(piece)
            pieces = clean
            for index, piece in enumerate(pieces):
                if piece in {'ing', '-ing'}:
                    logger.warning('word split: {} {}'.format(pieces[index - 1], piece))
            text.append(' '.join(pieces))

    corpus = [item.split() for item in text]
    min_count = 175
    size_word2vec = 1000
    iter_word2vec = 2
    random_state = 1
    model = Word2Vec(corpus,
                     # batch_words=20,
                     iter=iter_word2vec,
                     min_count=min_count,
                     seed=random_state,
                     size=size_word2vec, window=40,
                     workers=4,
                     )

    labels = [word for word in model.wv.vocab]
    tokens = [model.wv[word] for word in model.wv.vocab]
    logger.info('tokens with capitals: {}'.format(sorted([item for item in labels if str(item) != str(item).lower()])))
    logger.info('tokens all capitals: {}'.format(sorted([item for item in labels if str(item).isupper()])))

    tsne_model = TSNE(angle=0.5, early_exaggeration=12.0, init='pca', learning_rate=200.0,
                      method='barnes_hut',
                      # method='exact',
                      metric='euclidean', min_grad_norm=1e-07, n_components=2, n_iter=10000,
                      n_iter_without_progress=300,
                      perplexity=40.0, random_state=random_state, verbose=1, )
    tsne_values = tsne_model.fit_transform(tokens)

    xs = [value[0] for value in tsne_values]
    ys = [value[1] for value in tsne_values]

    plot_approach = 'plotly'
    if plot_approach == 'matplotlib':
        plt.figure(figsize=(16, 16))
        for i in range(len(tsne_values)):
            plt.scatter(xs[i], ys[i])
            plt.annotate(labels[i], ha='right', textcoords='offset points', va='bottom', xy=(xs[i], ys[i]),
                         xytext=(5, 2), )
        plt.tick_params(axis='both', bottom=False, labelbottom=False, labelleft=False, left=False, right=False,
                        top=False, which='both', )
        plt.show()
    elif plot_approach == 'plotly':
        colormap = 'jet'
        vectorizer = CountVectorizer(lowercase=False)
        fit_result = vectorizer.fit_transform(text)
        result = dict(zip(vectorizer.get_feature_names(), fit_result.toarray().sum(axis=0)))
        result = {key: int(result[key]) for key in result.keys() if str(key) in labels}
        misses = [item for item in labels if item not in result.keys()]
        logger.warning('label misses: {}'.format(misses))
        labels = [item for item in labels if item in result.keys()]
        color_index_map = {item: index for index, item in enumerate(sorted(set(result.values())))}
        colors = [float_color_to_hex(int(255.0 * float(color_index_map[result[this]]) / float(len(color_index_map))),
                                     cm.get_cmap(colormap)) for this in labels]

        figure = Figure(Scatter(hoverinfo='text', hovertext=['{}: {}'.format(item, result[item], ) for item in labels],
                                mode='text', text=labels, textfont=dict(color=colors, size=16, ), x=xs, y=ys, ),
                        layout=Layout(autosize=True, xaxis=dict(showticklabels=False),
                                      yaxis=dict(showticklabels=False), ))
        logger.info('saving HTML figure to {}'.format(output_file))
        plot(auto_open=False, auto_play=False, figure_or_data=figure, filename=output_file,
             link_text='', output_type='file', show_link=False, validate=True, )
    else:
        raise ValueError('plotting approach is {}. Quitting.'.format(plot_approach))
    logger.info('total time: {:5.2f}s'.format(time() - time_start))
