# modified from https://radimrehurek.com/gensim/models/doc2vec.html
import logging
from math import acos
from math import pi
from time import time

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import common_texts
from gensim.test.utils import get_tmpfile
from sklearn.metrics.pairwise import cosine_similarity

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
    # only do this if we're done training (i.e. we are not doing incremental training)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    similarity = cosine_similarity(model.infer_vector('system response'.split()).reshape(1, -1),
                                   model.infer_vector('stimulus response'.split()).reshape(1, -1), )

    logger.info(acos(similarity) / pi)

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
