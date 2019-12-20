from collections import Counter
from glob import glob
from json import load as json_load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

import matplotlib.pyplot as plt
from tika import parser
from wordcloud import WordCloud

if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started')

    with open('./demo_wordcloud.json', 'r') as settings_fp:
        settings = json_load(settings_fp, cls=None, object_hook=None, parse_float=None, parse_int=None,
                             parse_constant=None, object_pairs_hook=None)
    input_folder = settings['input_folder'] if 'input_folder' in settings.keys() else None
    if input_folder is None:
        logger.warning('input_folder is None. Quitting.')
        quit(code=1)
    stop_word = settings['stop_word'] if 'stop_word' in settings.keys() else list()
    if not len(stop_word):
        logger.warning('stop word list not in settings; default is empty.')
    token_count = settings['token_count'] if 'token_count' in settings.keys() else 10
    if 'token_count' not in settings.keys():
        logger.warning('token_count not in settings; default value is {}.'.format(token_count))
    imshow_interpolation = settings['imshow_interpolation'] if 'imshow_interpolation' in settings.keys() else 20
    if 'imshow_interpolation' not in settings.keys():
        logger.warning('imshow interpolation not in settings; default value is {}'.format(imshow_interpolation))

    result = list()
    input_file_count = 0
    for input_file_index, input_file in enumerate(glob(input_folder + '*.pdf')):
        logger.info(input_file)
        input_file_count += 1
        parse_result = parser.from_file(input_file)
        result.append(parse_result['content'])

    logger.info('file count: {}'.format(input_file_count))
    logger.info('result size: {}'.format(len(result)))

    logger.info('stop words: {}'.format(sorted(stop_word)))
    stop_word = set(stop_word)
    count = Counter()
    tokens_to_show = list()
    for item_index, item in enumerate(result):
        if item is not None:
            logger.info('item: {} size: {}'.format(item_index, len(item)))
            pieces = item.split()
            pieces = [piece.strip() for piece in pieces]
            pieces = [piece if not piece.endswith(':') else piece[:-1] for piece in pieces]
            for piece in pieces:
                count[piece] += 1 if all([
                    len(piece) > 1,
                    not piece.isdigit(),
                    piece.lower() not in stop_word,
                ]) else 0
            tokens_to_show = [token[0] for token in count.most_common(n=token_count)]
            logger.info(tokens_to_show[:20])
            logger.info(tokens_to_show[20:40])
            logger.info(tokens_to_show[40:])
        else:
            logger.info('item: {} size: 0 '.format(item_index))

    to_show = {count_item[0]: count_item[1] for count_item in count.most_common(n=token_count)}

    word_cloud = WordCloud().generate_from_frequencies(frequencies=to_show, max_font_size=30)

    plt.imshow(word_cloud, interpolation=imshow_interpolation)
    plt.axis('off')
    plt.show()

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
