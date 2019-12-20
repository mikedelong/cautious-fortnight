from collections import Counter
from glob import glob
from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

from tika import parser
from wordcloud import WordCloud

input_folder = './afghanistan-papers-documents/'
token_count = 40
if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started')

    result = list()
    input_file_count = 0
    for input_file_index, input_file in enumerate(glob(input_folder + '*.pdf')):
        logger.info(input_file)
        input_file_count += 1
        parse_result = parser.from_file(input_file)
        result.append(parse_result['content'])

    logger.info('file count: {}'.format(input_file_count))
    logger.info('result size: {}'.format(len(result)))

    stop_word = ['-', '1', '2', 'a', 'about', 'all', 'also', 'an', 'and', 'are', 'as', 'at', 'be', 'because', 'been',
                 'but', 'by', 'can', 'could', 'did', "didn't", 'do', "don't", 'for', 'from', 'get', 'going', 'good',
                 'had', 'has', 'have', 'he', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'just', 'like', 'more', 'need',
                 'no', 'not', 'of', 'on', 'one', 'or', 'other', 'our', 'out', 'should', 'so', 'some', 'that', 'the',
                 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'time', 'to', 'up', 'very', 'was', 'we',
                 'were', 'what', 'when', 'which', 'who', 'will', 'with', 'work', 'would', 'you', '•', '4', '11',
                 '--', 'go', 't', 'where', 'than', '3', 'know', 'want', 'much', 'only', '.', 'lot', 'wanted',
                 'it.', 'got', 'make', 'things', 'back', 'through', 'over', 'doing', 'even', '5', 'any', 'his',
                 'doing', 'even', '6', 's', 'know,', '10', 'way', 'said', 'its', 'came', 'us', 'between', 'take',
                 'those', 'really', 'R.', 'many', 'used', 'never', 'interview', 'dr.',
                 '7', 'after', 'it\'s', 'see', 'my', 'made', 'went', ]

    logger.info('stop words: {}'.format(sorted(stop_word)))
    stop_word = set(stop_word)
    count = Counter()
    for item_index, item in enumerate(result):
        if item is not None:
            logger.info('item: {} size: {}'.format(item_index, len(item)))
            pieces = item.split()
            pieces = [piece.strip() for piece in pieces]
            for piece in pieces:
                count[piece] += 1 if piece.lower() not in stop_word else 0
            tokens_to_show = [token[0] for token in count.most_common(n=token_count)]
            logger.info(tokens_to_show[:20])
            logger.info(tokens_to_show[20:])
        else:
            logger.info('item: {} size: 0 '.format(item_index))

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
