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

    input_files = [input_file for input_file in glob(input_folder + '*.pdf')]
    items = list()
    for input_file in input_files:
        parse_result = parser.from_file(input_file)
        if parse_result['content']:
            items.append(unidecode(parse_result['content']))
            logger.info('length: {} name: {}'.format(len(parse_result['content']), input_file.replace('\\', '/')))
        else:
            logger.warning('length: 0 name: {}'.format(input_file))
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
    # todo factor these out as data
    split = {'U.S.': ['US'], 'U.S': ['US'], 'Minh\'s': ['Minh'], 'President\'s': ['President'], 'Ho\'s': ['Ho'],
             'Amembassy': ['American', 'Embassy'], 'Apr': ['April'], 'Jan': ['January'], 'Feb': ['February'],
             'Mar': ['March'], 'Jun': ['June'], 'Jul': ['July'], 'Aug': ['August'], 'Sep': ['September'],
             'Oct': ['October'], 'Nov': ['November'], 'Dec': ['December'], 'budgetconsultation':
                 ['budget', 'consultation'], 'Msg': ['message'], 'msg': ['message'], 'Diem\'s': ['Diem'],
             'Viet-Nam': ['Vietnam'], 'Vietnem': ['Vietnam'], 'deg': ['degree'], }
    # todo factor these out as data
    join = {'Accord-', 'Act-', 'accept-', 'accord-', 'act-', 'arm-', 'attach-', 'attack-', 'back-', 'bargain-',
            'belong-', 'bind-', 'boil-', 'bomb', 'bomb-', 'bowl', 'carry-', 'choos-', 'command-', 'compromis-',
            'concern-', 'contemplat-', 'continu-', 'cover-', 'declin-', 'demand-', 'develop-', 'dur-', 'emanat-',
            'emerg-', 'establish-', 'even-', 'exclud-', 'exist-', 'express-', 'fight-', 'follow-', 'group-', 'grow-',
            'hear-', 'imprison-', 'improv-', 'includ-', 'increas-', 'inject-', 'interest-', 'issu-', 'liv-',
            'maintain-', 'mak-', 'mean-', 'meet-', 'morn-', 'negotiat-', 'organiz-', 'prevail-', 'proceed-', 'propos-',
            'provid-', 'question-', 'reach-', 'reconven-', 'regard-', 'report-', 'return-', 'secur-', 'seek-', 'start-',
            'stress-', 'strik-', 'support-', 'tak-', 'train-', 'understand-', 'visit-', 'warn-', 'will-', 'work-',
            'lead-', 'demonstrat-', 'bring-', 'involv-', 'operat-', 'encourag-', 'help-', 'resist-', 'expand-',
            'regroup-', 'achiev-', 'unyield-', 'open-', 'try-', 'forward-', 'speak-', 'show-', 'Follow-',
            'send-', 'urg-', 'Brief-', 'limit-', 'mobiliz-', 'consider-', 'surround-', 'present-', 'weaken-',
            'mount-', 'effec-', 'alterna-', 'objec-', 'representa-', 'collec-', 'Sensi', 'sensi-', 'initia-',
            'administra-', 'nega-', 'posi-', 'ineffec-', 'construc-', 'coopera-', 'direc-', }
    logger.info('join data: {}'.format(sorted(list(join))))
    # todo factor these out as data
    joined = {'According', 'Acting', 'accepting', 'according', 'acting', 'arming', 'attaching', 'attacking', 'backing',
              'bargaining', 'belonging', 'binding', 'boiling', 'bombing', 'bowling', 'carrying', 'choosing',
              'commanding', 'compromising', 'concerning', 'contemplating', 'continuing', 'covering', 'declining',
              'demanding', 'developing', 'during', 'emanating', 'emerging', 'establishing', 'evening', 'excluding',
              'existing', 'expressing', 'fighting', 'following', 'grouping', 'growing', 'hearing', 'imprisoning',
              'improving', 'including', 'increasing', 'injecting', 'interesting', 'issuing', 'living', 'maintaining',
              'making', 'meaning', 'meeting', 'morning', 'negotiating', 'organizing', 'prevailing', 'proceeding',
              'proposing', 'providing', 'questioning', 'reaching', 'reconvening', 'regarding', 'reporting', 'returning',
              'securing', 'seeking', 'starting', 'stressing', 'striking', 'supporting', 'taking', 'training',
              'understanding', 'visiting', 'warning', 'willing', 'working', 'leading', 'demonstrating', 'involving',
              'operating', 'encouraging', 'helping', 'resisting', 'expanding', 'regrouping', 'achieving', 'bringing',
              'unyielding', 'opening', 'trying', 'forwarding', 'speaking', 'showing', 'Following', 'sending', 'urging',
              'Briefing', 'limiting', 'mobilizing', 'considering', 'surrounding', 'presenting', 'weakening',
              'mounting', 'effective', 'alternative', 'objective', 'representative', 'collective', 'Sensitive',
              'initiative', 'administrative', 'sensitive', 'negative', 'positive', 'ineffective', 'constructive',
              'cooperative', 'directive', }
    logger.info('joined data: {}'.format(sorted(list(joined))))

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
                if piece in {'ing', 'tive'}:
                    if pieces[index - 1] not in joined:
                        clean.append(piece)
                else:
                    clean.append(piece)
            pieces = clean
            for index, piece in enumerate(pieces):
                if piece in {'ing', 'tive'}:
                    logger.warning('word split: {} {}'.format(pieces[index - 1], piece))
                    ing_counts.update({pieces[index - 1]: 1})
            text.append(' '.join(pieces))

    logger.info(ing_counts.most_common(n=20))
    corpus = [item.split() for item in text]
    min_count = 175
    # size_word2vec = 1000
    size_word2vec = 100
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
    logger.info('words with lowercase cognates: {}'.format(
        [item for item in labels if item != item.lower() and item.lower() in labels]))
    tokens = [model.wv[word] for word in model.wv.vocab]
    logger.info('tokens with capitals: {}'.format(sorted([item for item in labels if str(item) != str(item).lower()])))
    logger.info('tokens all capitals: {}'.format(sorted([item for item in labels if str(item).isupper()])))

    # learning_rate = 200.0
    # learning_rate = 40.0
    learning_rate = 1000.0
    # min_grad_norm = 1e-7
    min_grad_norm = 1e-7
    # perplexity = 40.0
    # perplexity = 9.0
    perplexity = 10.0
    n_iter = 35000
    tsne_init = 'pca'  # 'pca'
    tsne_model = TSNE(angle=0.5, early_exaggeration=12.0,
                      init=tsne_init, learning_rate=learning_rate,
                      method='barnes_hut',
                      # method='exact',
                      metric='euclidean', min_grad_norm=min_grad_norm, n_components=2, n_iter=n_iter,
                      n_iter_without_progress=300,
                      perplexity=perplexity, random_state=random_state, verbose=1, )
    tsne_values = tsne_model.fit_transform(tokens)
    logger.info('TSNE completes after {} iterations of {} allowed'.format(tsne_model.n_iter_, n_iter))

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

    # calculate some similarities
    top_n = 10
    for word in sorted(['draft', 'Draft', 'secret', 'Secretary', 'one', 'two', 'three', 'four', 'five', 'six',
                        'Dept', 'Department', 'USSR', 'Admiral', 'Armed', 'November', 'April', 'June', 'July',
                        'May', 'Vietnam', ]):
        logger.info('most similar to {}: {}'.format(word, model.wv.most_similar(word, topn=top_n)))
    logger.info('China/Peking similarity: {}'.format(model.wv.similarity('China', 'Peking')))
    logger.info('Dept/Department similarity: {}'.format(model.wv.similarity('Dept', 'Department')))
    logger.info('draft/Draft similarity: {}'.format(model.wv.similarity('draft', 'Draft')))
    logger.info('Eisenhower/Kennedy similarity: {}'.format(model.wv.similarity('Eisenhower', 'Kennedy')))
    logger.info('June/July similarity: {}'.format(model.wv.similarity('June', 'July')))
    logger.info('one/One similarity: {}'.format(model.wv.similarity('one', 'One')))
    logger.info('secret/Secretary similarity: {}'.format(model.wv.similarity('secret', 'Secretary')))
    logger.info('south/South similarity: {}'.format(model.wv.similarity('south', 'South')))

    for word in [item for item in labels if item != item.lower() and item.lower() in labels]:
        logger.info('{}/{} similarity: {}'.format(word.lower(), word, model.wv.similarity(word.lower(), word)))
    logger.info('total time: {:5.2f}s'.format(time() - time_start))
