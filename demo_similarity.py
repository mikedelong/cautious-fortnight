import logging
from collections import defaultdict
from time import time

from gensim import corpora
from gensim.similarities.docsim import Similarity

context_limit_ = 1000
input_file = './data/35830.txt'
text_start = 2124
text_stop = 524200

if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info('started')

    with open(input_file, 'r') as input_fp:
        text = input_fp.read()
        text = text.split('\n')
        text = text[text_start:text_stop]
        text = ' '.join(text)
        logger.info('text length: {}'.format(len(text)))
        pieces = [text[i:i + context_limit_] for i in range(0, len(text), context_limit_)] + [
            text[i + context_limit_ // 2: i + 3 * context_limit_ // 2] for i in
            range(0, len(text) - context_limit_, context_limit_)]
        logger.info('context size: {} pieces: {}'.format(context_limit_, len(pieces)))

        # remove common words and tokenize
        stoplist = set('for a of the and to in'.split())
        texts = [[word for word in document.lower().split() if word not in stoplist] for document in pieces]

        # remove words that appear only once
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1

        texts = [[token for token in text if frequency[token] > 1] for text in texts]

        dictionary = corpora.Dictionary(texts)
        corpus_ = [dictionary.doc2bow(text) for text in texts]

    index = Similarity(output_prefix='t_', corpus=corpus_, num_features=1000)

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
