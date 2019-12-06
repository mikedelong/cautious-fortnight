import logging
from time import time

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import common_texts
from gensim.test.utils import get_tmpfile

if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info('started')

    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
    # note this goes in our temporary file directory
    file_name = get_tmpfile('demo_doc2vec_model.gensim')
    model.save(file_name)
    model = Doc2Vec.load(file_name)

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
