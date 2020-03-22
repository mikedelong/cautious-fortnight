from collections import Counter
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
from sklearn.decomposition import PCA
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


def shorten_similarity(arg):
    return [(item[0], round(item[1], 3)) for item in arg]


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
    join_targets = settings['join_target'] if 'join_target' in settings.keys() else None
    if join_targets:
        logger.info('join target file file: {}'.format(join_targets))
    else:
        logger.warning('join target file is missing.')
    join_fixes = settings['join_fix'] if 'join_fix' in settings.keys() else None
    if join_fixes:
        logger.info('join fix file file: {}'.format(join_fixes))
    else:
        logger.warning('join fix file is missing.')

    lowercase_fixes = settings['lowercase_fixes'] if 'lowercase_fixes' in settings.keys() else None
    if lowercase_fixes:
        logger.info('lowercase fix file: {}'.format(lowercase_fixes))
    else:
        logger.warning('lower case fix file is missing.')
    output_file = settings['output_file'] if 'output_file' in settings.keys() else None
    if output_file:
        logger.info('output file: {}'.format(output_file))
    else:
        logger.warning('output file is missing from the settings. Quitting.')
        quit(code=2)
    proper_name_fixes = settings['proper_name_fixes'] if 'proper_name_fixes' in settings.keys() else None
    if proper_name_fixes:
        logger.info('proper name fix file: {}'.format(proper_name_fixes))
    else:
        logger.warning('proper name fix file is missing.')
    split_fixes = settings['split_fixes'] if 'split_fixes' in settings.keys() else None
    if split_fixes:
        logger.info('split fix file: {}'.format(split_fixes))
    else:
        logger.warning('split fix file is missing.')

    # get the data for the various lexical fixes
    if join_fixes:
        with open(join_fixes, 'r') as join_fix_fp:
            join_fix_data = json_load(join_fix_fp)
            joined = set(join_fix_data['data'])
    logger.info('join fix data: {}'.format(sorted(list(joined))))
    if join_targets:
        with open(join_targets, 'r') as join_fp:
            join_data = json_load(join_fp)
            join = set(join_data['data'])
    logger.info('join target data: {}'.format(sorted(list(join))))

    if lowercase_fixes:
        with open(lowercase_fixes, 'r') as lowercase_fp:
            lowercase_data = json_load(lowercase_fp)
            capitalization = set(lowercase_data['data'])
    logger.info('capitalization tokens: {}'.format(sorted(list(capitalization))))
    if proper_name_fixes:
        with open(proper_name_fixes, 'r') as proper_fp:
            proper_name_data = json_load(proper_fp)
            proper = set(proper_name_data['data'])
    logger.info('proper noun fixes: {}'.format(sorted(list(proper))))
    if split_fixes:
        with open(split_fixes, 'r') as split_fixes_fp:
            split_fix_data = json_load(split_fixes_fp)
            split = split_fix_data['data']
    logger.info('split fixes: {}'.format(split))

    # read and process the input PDF files
    input_files = [input_file for input_file in glob(input_folder + '*.pdf')]
    items = list()
    for input_file in input_files:
        parse_result = parser.from_file(input_file)
        if parse_result['content']:
            items.append(unidecode(parse_result['content']))
            logger.info('length: {} name: {}'.format(len(parse_result['content']), input_file.replace('\\', '/')))
        else:
            logger.warning('length: 0 name: {}'.format(input_file))

    ing_counts = Counter()
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
                if piece in {'ing', 'tion', 'tive'}:
                    if pieces[index - 1] not in joined:
                        clean.append(piece)
                else:
                    clean.append(piece)
            pieces = clean
            for index, piece in enumerate(pieces):
                if piece in {'ing', 'tion', 'tive'}:
                    logger.warning('word split: {} {}'.format(pieces[index - 1], piece))
                    ing_counts.update({pieces[index - 1]: 1})
            text.append(' '.join(pieces))

    logger.info(
        [item for item in list(ing_counts.most_common(n=25)) if not item[0] in {'of', 'to', 'the', 'and', 'is'}])
    corpus = [item.split() for item in text]
    random_state = 3
    word2vec_batch_words = False
    word2vec_count = [100, 175][1]
    word2vec_size = [50, 100, 200, 300][2]
    word2vec_iterations = 2
    word2vec_window_size = 40
    word2vec_worker_count = 4
    model = Word2Vec(corpus, batch_words=word2vec_batch_words, iter=word2vec_iterations, min_count=word2vec_count,
                     seed=random_state, size=word2vec_size, window=word2vec_window_size,
                     workers=word2vec_worker_count, )

    labels = [word for word in model.wv.vocab]
    logger.info('words with lowercase cognates: {}'.format(
        [item for item in labels if item != item.lower() and item.lower() in labels]))
    tokens = [model.wv[word] for word in model.wv.vocab]
    logger.info('tokens with capitals: {}'.format(sorted([item for item in labels if str(item) != str(item).lower()])))
    logger.info('tokens all capitals: {}'.format(sorted([item for item in labels if str(item).isupper()])))

    visualization = 'pca'
    if visualization == 'tsne':
        tsne_angle = 0.5
        tsne_early_exaggeration = 12.0
        tsne_init = ['pca', 'random'][0]
        tsne_learning_rate = 1000.0
        tsne_method = ['barnes_hut', 'exact'][0]
        tsne_metric = ['cosine', 'euclidean'][0]
        tsne_min_grad_norm = [1e-7][0]
        tsne_n_components = 2
        tsne_n_iter = 35000
        tsne_n_iter_without_process = 300
        tsne_perplexity = [9.0, 10.0, 40.0][1]
        tsne_verbosity = 1
        tsne_model = TSNE(angle=tsne_angle, early_exaggeration=tsne_early_exaggeration, init=tsne_init,
                          learning_rate=tsne_learning_rate, method=tsne_method, metric=tsne_metric,
                          min_grad_norm=tsne_min_grad_norm, n_components=tsne_n_components, n_iter=tsne_n_iter,
                          n_iter_without_progress=tsne_n_iter_without_process, perplexity=tsne_perplexity,
                          random_state=random_state, verbose=tsne_verbosity, )
        values = tsne_model.fit_transform(tokens)
        logger.info('TSNE completes after {} iterations of {} allowed'.format(tsne_model.n_iter_, tsne_n_iter))
    elif visualization == 'pca':
        pca_model = PCA(n_components=2, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto',
                        random_state=random_state)
        values = pca_model.fit_transform(tokens)
    else:
        raise NotImplementedError('visualization must be t-SNE or PCA.')

    xs = [value[0] for value in values]
    ys = [value[1] for value in values]

    plot_approach = 'plotly'
    if plot_approach == 'matplotlib':
        plt.figure(figsize=(16, 16))
        for i in range(len(values)):
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

    # calculate some similarities
    top_n = 10
    for word in sorted(['draft', 'Draft', 'secret', 'Secretary', 'one', 'two', 'three', 'four', 'five', 'six',
                        'Dept', 'Department', 'USSR', 'Admiral', 'Armed', 'November', 'April', 'June', 'July',
                        'May', 'Vietnam', ]):
        logger.info('most similar to {}: {}'.format(word, shorten_similarity(model.wv.most_similar(word, topn=top_n))))
    logger.info('China/Peking similarity: {:5.3f}'.format(model.wv.similarity('China', 'Peking')))
    logger.info('Dept/Department similarity: {:5.3f}'.format(model.wv.similarity('Dept', 'Department')))
    logger.info('Eisenhower/Kennedy similarity: {:5.3f}'.format(model.wv.similarity('Eisenhower', 'Kennedy')))
    logger.info('June/July similarity: {:5.3f}'.format(model.wv.similarity('June', 'July')))
    logger.info('secret/Secretary similarity: {:5.3f}'.format(model.wv.similarity('secret', 'Secretary')))
    logger.info('tho/though similarity: {:5.3f}'.format(model.wv.similarity('secret', 'Secretary')))

    for word in sorted([item for item in labels if item != item.lower() and item.lower() in labels]):
        logger.info('{}/{} similarity: {:5.3f}'.format(word.lower(), word, model.wv.similarity(word.lower(), word)))
    logger.info('total time: {:5.2f}s'.format(time() - time_start))
