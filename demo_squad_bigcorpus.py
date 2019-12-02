import logging
from collections import Counter
from random import choice
from time import time

from deeppavlov import build_model
from gensim import corpora
from gensim import models
from gensim.similarities.docsim import MatrixSimilarity
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

context_limit_ = 1000
configuration = {
    'chainer': {
        'in': ['context_raw', 'question_raw'],
        'in_y': ['ans_raw', 'ans_raw_start'],
        'pipe': [{'char_limit': 16,
                  'class_name': 'squad_preprocessor',
                  'context_limit': context_limit_,  # was 400
                  'id': 'squad_prepr',
                  'in': ['context_raw', 'question_raw'],
                  'out': ['context', 'context_tokens', 'context_chars', 'c_r2p', 'c_p2r', 'question', 'question_tokens',
                          'question_chars', 'spans'],
                  'question_limit': 150, },
                 {'class_name': 'squad_ans_preprocessor',
                  'id': 'squad_ans_prepr',
                  'in': ['ans_raw', 'ans_raw_start', 'c_r2p', 'spans'],
                  'out': ['ans', 'ans_start', 'ans_end'], },
                 {'char_limit': '#squad_prepr.char_limit',
                  'class_name': 'squad_vocab_embedder',
                  'context_limit': '#squad_prepr.context_limit',
                  'emb_folder': '{DOWNLOADS_PATH}/embeddings/',
                  'emb_url': 'http://files.deeppavlov.ai/embeddings/wiki-news-300d-1M.vec',
                  'fit_on': ['context_tokens', 'question_tokens'],
                  'id': 'vocab_embedder',
                  'in': ['context_tokens', 'question_tokens'],
                  'level': 'token',
                  'load_path': '{MODELS_PATH}/squad_model/emb/vocab_embedder.pckl',
                  'out': ['context_tokens_idxs', 'question_tokens_idxs'],
                  'question_limit': '#squad_prepr.question_limit',
                  'save_path': '{MODELS_PATH}/squad_model/emb/vocab_embedder.pckl', },
                 {'char_limit': '#squad_prepr.char_limit',
                  'class_name': 'squad_vocab_embedder',
                  'context_limit': '#squad_prepr.context_limit',
                  'emb_folder': '{DOWNLOADS_PATH}/embeddings/',
                  'emb_url': 'http://files.deeppavlov.ai/embeddings/wiki-news-300d-1M-char.vec',
                  'fit_on': ['context_chars', 'question_chars'],
                  'id': 'char_vocab_embedder',
                  'in': ['context_chars', 'question_chars'],
                  'level': 'char',
                  'load_path': '{MODELS_PATH}/squad_model/emb/char_vocab_embedder.pckl',
                  'out': ['context_chars_idxs', 'question_chars_idxs'],
                  'question_limit': '#squad_prepr.question_limit',
                  'save_path': '{MODELS_PATH}/squad_model/emb/char_vocab_embedder.pckl', },
                 {'attention_hidden_size': 75,
                  'char_emb': '#char_vocab_embedder.emb_mat',
                  'char_hidden_size': 100,
                  'char_limit': '#squad_prepr.char_limit',
                  'class_name': 'squad_model',
                  'clip_norm': 5.0,
                  'context_limit': '#squad_prepr.context_limit',
                  'encoder_hidden_size': 75,
                  'id': 'squad',
                  'keep_prob': 0.7,
                  'learning_rate': 0.5,
                  'learning_rate_drop_div': 2.0,
                  'learning_rate_drop_patience': 5,
                  'load_path': '{MODELS_PATH}/squad_model/model',
                  'in': ['context_tokens_idxs', 'context_chars_idxs', 'question_tokens_idxs', 'question_chars_idxs'],
                  'in_y': ['ans_start', 'ans_end'],
                  'min_learning_rate': 0.001,
                  'momentum': 0.95,
                  'optimizer': 'tf.train:AdadeltaOptimizer',
                  'out': ['ans_start_predicted', 'ans_end_predicted', 'logits'],
                  'question_limit': '#squad_prepr.question_limit',
                  'save_path': '{MODELS_PATH}/squad_model/model',
                  'train_char_emb': True,
                  'word_emb': '#vocab_embedder.emb_mat', },
                 {'class_name': 'squad_ans_postprocessor',
                  'id': 'squad_ans_postprepr',
                  'in': ['ans_start_predicted', 'ans_end_predicted', 'context_raw', 'c_p2r', 'spans'],
                  'out': ['ans_predicted', 'ans_start_predicted', 'ans_end_predicted'], }],
        'out': ['ans_predicted', 'ans_start_predicted', 'logits'],
    },
    'dataset_iterator': {'class_name': 'squad_iterator', 'seed': 1337, 'shuffle': True, },
    'dataset_reader': {'class_name': 'squad_dataset_reader', 'data_path': '{DOWNLOADS_PATH}/squad/', },
    'metadata': {
        'variables': {
            'ROOT_PATH': '~/.deeppavlov',
            'DOWNLOADS_PATH': '{ROOT_PATH}/downloads',
            'MODELS_PATH': '{ROOT_PATH}/models',
        },
        'download': [{'subdir': '{MODELS_PATH}',
                      'url': 'http://files.deeppavlov.ai/deeppavlov_data/squad_model_1.4_cpu_compatible.tar.gz', },
                     {'subdir': '{DOWNLOADS_PATH}/embeddings',
                      'url': 'http://files.deeppavlov.ai/embeddings/wiki-news-300d-1M.vec', },
                     {'subdir': '{DOWNLOADS_PATH}/embeddings',
                      'url': 'http://files.deeppavlov.ai/embeddings/wiki-news-300d-1M-char.vec', }],
        'labels': {'server_utils': 'SquadModel', 'telegram_utils': 'SquadModel', },
        'requirements': ['{DEEPPAVLOV_PATH}/requirements/tf.txt', ],
    },
    'train': {'batch_size': 50, 'class_name': 'nn_trainer', 'evaluation_targets': ['valid'], 'log_every_n_batches': 250,
              'metrics': [{'inputs': ['ans_raw', 'ans_predicted'], 'name': 'squad_v1_em', },
                          {'inputs': ['ans_raw', 'ans_predicted'], 'name': 'squad_v1_f1', }, ], 'pytest_max_batches': 2,
              'show_examples': False, 'val_every_n_epochs': 1, 'validation_patience': 10, },
}
exit_questions = {'bye', 'cya', 'exit', 'good-bye', 'good-by', 'quit'}
input_file = './data/35830.txt'
lsi_topic_count = 300
miss_responses = ['I don\'t know anything about that.', 'No clue.', 'Reply hazy. Please try again.']
modes = ['cosine_similarity', 'lsi_similarity', ]
mode = modes[0]
results_to_return = 7
similarity_feature_count = 300
text_start = 2124
text_stop = 524200

if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info('started')

    model = build_model(configuration, download=True)
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
        if mode == modes[0]:
            vectorizer = TfidfVectorizer()
            vectorizer.fit(pieces)
            pieces_ = vectorizer.transform(pieces)
        elif mode == modes[1]:
            texts = [[word for word in document.lower().split() if word not in ENGLISH_STOP_WORDS] for document in
                     pieces]
            # remove words that appear only once
            frequency = Counter([token for text in texts for token in text])
            texts = [[token for token in text if frequency[token] > 1] for text in texts]
            dictionary = corpora.Dictionary(texts)
            logger.info('dictionary size: {}'.format(len(dictionary)))
            corpus_ = [dictionary.doc2bow(text) for text in texts]
            lsi = models.LsiModel(corpus_, id2word=dictionary, num_topics=lsi_topic_count)
            matrix_similarity = MatrixSimilarity(lsi[corpus_], num_features=similarity_feature_count)
        else:
            raise ValueError('mode can only be {} but is [{}]'.format(modes, mode))

    logger.info('ready.')

    done = False
    while not done:
        question = input('?: ')
        if question.lower() not in exit_questions:
            if mode == modes[0]:
                question_ = vectorizer.transform([question])
                cosine_similarities = cosine_similarity(question_, pieces_).flatten()
                related_product_indices = cosine_similarities.argsort()[:-results_to_return - 1:-1]
                if cosine_similarity(question_, pieces_[related_product_indices[0]])[0][0] == 0.0:
                    logging.info('Q: {} : cos: {} A: {}'.format(question, 0.0, choice(miss_responses)))
                else:
                    for index in related_product_indices:
                        result = model([pieces[index]], [question])
                        logger.info('Q: {} cos: {:5.3f} A: {}'.format(question,
                                                                      cosine_similarity(question_,
                                                                                        pieces_[index])[0][0],
                                                                      result[0]))
            elif mode == modes[1]:
                question_ = lsi[dictionary.doc2bow(question.lower().split())]
                similarities = sorted(enumerate(matrix_similarity[question_]), key=lambda item: -item[1])[
                               :results_to_return]
                if similarities[0][1] == 0.0:
                    logging.info('Q: {} : lsi: {} A: {}'.format(question, 0.0, choice(miss_responses)))
                else:
                    for similarity in similarities:
                        result = model([pieces[similarity[0]]], [question])
                        logging.info('Q: {} : lsi: {} A: {}'.format(question, similarity, result[0]))
            else:
                raise ValueError('mode can only be {} but is [{}]'.format(modes, mode))
        else:
            done = True

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
