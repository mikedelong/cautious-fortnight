# modified from https://radimrehurek.com/gensim/models/doc2vec.html
import logging
from math import acos
from math import pi
from time import time

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile
from sklearn.metrics.pairwise import cosine_similarity

do_build_model = False
file_name = get_tmpfile('demo_doc2vec_model.gensim')
fruit_flies = 'human interface time'  # 'Fruit flies like an apple.'
time_flies = 'computer user survey'  # 'Time flies like an arrow.'
two_over_pi = 2.0 / pi

raw_documents = [
    'human interface computer',
    'survey user computer system response time',
    'eps user interface system',
    'system human system eps',
    'user response time',
    'trees',
    'graph trees',
    'graph minors trees',
    'graph minors survey'
]

if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info('started')

    if do_build_model:
        documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(raw_documents)]
        for document in documents:
            logger.info(document)
        model = Doc2Vec(documents, vector_size=5, window=3, min_count=1, workers=4, seed=1)
        # note this goes in our temporary file directory
        model.save(file_name)
        # only do this if we're done training (i.e. we are not doing incremental training)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model = Doc2Vec.load(file_name)

    fruit_flies_ = model.infer_vector(fruit_flies.split())
    logger.info(fruit_flies_)
    time_flies_ = model.infer_vector(time_flies.split())
    logger.info(time_flies_)
    similarity = cosine_similarity(fruit_flies_.reshape(1, -1), time_flies_.reshape(1, -1), )

    logger.info('angular similarity: {:5.3f}'.format(1.0 - two_over_pi * acos(similarity)))

    for count in range(10):
        logger.info('{} {}'.format(count, model.infer_vector(fruit_flies.split())))

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
