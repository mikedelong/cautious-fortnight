# modified from https://radimrehurek.com/gensim/models/doc2vec.html
from logging import INFO
from logging import basicConfig
from logging import getLogger
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


do_build_model = True
file_name = get_tmpfile('demo_doc2vec_model.gensim')
scenario = 0

if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)

    logger.info('started')

    if scenario == 0:
        raw_documents = ['eps user interface system', 'graph minors survey', 'graph minors trees', 'graph trees',
                         'human interface computer', 'survey user computer system response time',
                         'system human system eps', 'trees', 'user response time']
        sentence = ['human interface time', 'computer user survey', ]
    elif scenario == 1:
        raw_documents = ['an apple is a kind of fruit', 'a banana is a kind of fruit',
                         'the apple never falls far from the tree', 'when life gives you an apple make sauce',
                         'you might find both an apple and a banana in the produce section of the grocery store',
                         'i do this all the time', 'time passes', 'i love her all the time',
                         'please do not waste my time', 'time is money', 'whenever he can he flies his plane',
                         'whenever he can he flies', 'i would like to help you if i can',
                         'close your mouth you will draw flies', 'you will catch more flies with honey',
                         'the outfielder caught pop flies', 'fly is singular, flies is plural',
                         'i don\'t know why she swallowed those flies', ]

        sentence = ['Fruit flies like an apple.', 'Time flies like an arrow.', ]
        sentence = [item.lower().replace('.', '') for item in sentence]

    else:
        raise ValueError('scenario must be either 0 or 1 but is instead {}. Quitting.'.format(scenario))

    if do_build_model:
        documents = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(raw_documents)]
        for document in documents:
            logger.info(document)
        model = Doc2Vec(documents, vector_size=10, window=3, min_count=1, workers=4, seed=1,
                        epochs=5000)
        # note this goes in our temporary file directory
        model.save(file_name)
        # only do this if we're done training (i.e. we are not doing incremental training)
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model = Doc2Vec.load(file_name)

    similarities = [get_angular_similarity(model, sentence[0], sentence[1]) for _ in range(10)]
    for index, current in enumerate(similarities):
        logger.info('{} {:5.4f} {:5.4f} {:5.4f} {:5.4f}'.format(index, current, min(similarities[:index + 1]),
                                                                mean(similarities[:index + 1]),
                                                                max(similarities[:index + 1])))

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
