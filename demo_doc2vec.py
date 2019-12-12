# modified from https://radimrehurek.com/gensim/models/doc2vec.html
import logging
from math import acos
from math import pi
from time import time

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile
from numpy import max
from numpy import mean
from numpy import min
from sklearn.metrics.pairwise import cosine_similarity


def get_angular_similarity(arg_model, arg_left, arg_right):
    epochs_ = 100
    left_ = arg_model.infer_vector(arg_left.split(), epochs=epochs_)
    right_ = arg_model.infer_vector(arg_right.split(), epochs=epochs_)
    similarity_ = cosine_similarity(left_.reshape(1, -1), right_.reshape(1, -1), )
    return 1.0 - acos(similarity_) / pi


do_build_model = False
file_name = get_tmpfile('demo_doc2vec_model.gensim')
t0 = 'human interface time'  #
t1 = 'computer user survey'  #

raw_documents = ['human interface computer', 'survey user computer system response time', 'eps user interface system',
                 'system human system eps', 'user response time', 'trees', 'graph trees', 'graph minors trees',
                 'graph minors survey']

other_raw_documents = ['an apple is a kind of fruit', 'a banana is a kind of fruit',
                       'the apple never falls far from the tree', 'when life gives you an apple make sauce',
                       'you might find both an apple and a banana in the produce section of the grocery store',
                       'i do this all the time', 'time passes', 'i love her all the time',
                       'please do not waste my time', 'time is money',
                       'whenever he can he flies his plane', 'whenever he can he flies',
                       'i would like to help you if i can', 'close your mouth you will draw flies',
                       'you will catch more flies with honey', 'the outfielder caught pop flies',
                       'fly is singular, flies is plural', 'i don\'t know why she swallowed those flies']
t2 = 'Fruit flies like an apple.'.lower().replace('.', '')
t3 = 'Time flies like an arrow.'.lower().replace('.', '')

if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info('started')

    if do_build_model:
        documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(other_raw_documents)]
        for document in documents:
            logger.info(document)
        model = Doc2Vec(documents, vector_size=10, window=3, min_count=1, workers=4, seed=1,
                        epochs=5000)
        # note this goes in our temporary file directory
        model.save(file_name)
        # only do this if we're done training (i.e. we are not doing incremental training)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model = Doc2Vec.load(file_name)

    similarities = list()
    for count in range(10):
        current = get_angular_similarity(model, t2, t3)
        similarities.append(current)
        logger.info('{} {:5.4f} {:5.4f} {:5.4f} {:5.4f}'.format(count, current, min(similarities),
                                                                mean(similarities), max(similarities)))

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
