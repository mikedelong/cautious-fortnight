from collections import Counter
from json import load as load_json
from logging import INFO
from logging import basicConfig
from logging import getLogger
from logging import info
from math import acos
from math import pi
from random import choice
from time import time

from deeppavlov import build_model
from gensim import corpora
from gensim import models
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.similarities.docsim import MatrixSimilarity
from gensim.summarization.textcleaner import split_sentences
from gensim.summarization.textcleaner import tokenize_by_word
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

configuration = {
    'chainer': {
        'in': ['context_raw', 'question_raw'],
        'in_y': ['ans_raw', 'ans_raw_start'],
        'pipe': [{'char_limit': 16,
                  'class_name': 'squad_preprocessor',
                  'context_limit': 'X',  # was 400
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
doc2vec_min_count = 1
doc2vec_question_epochs = 200
doc2vec_seed = 1
doc2vec_vector_size = 10
doc2vec_window = 3
doc2vec_workers = 4
exit_questions = {'bye', 'cya', 'exit', 'good-bye', 'good-by', 'quit'}
input_file = './data/35830.txt'
lsi_topic_count = 200
miss_responses = ['Ask again later.', 'I don\'t know anything about that.', 'No clue.', 'Reply hazy, Try again.']
modes = ['cosine_similarity', 'lsi_similarity', 'doc2vec_cosine', 'doc2vec_most_similar']
mode = modes[3]
pieces_strategies = ['character', 'sentence', ]
pieces_strategy = pieces_strategies[1]
results_to_return = 7
sentences_per_chunk = 10
similarity_feature_count = 200
text_start = 2124
text_stop = 524200

if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)

    logger.info('started.')

    with open('./demo_squad_bigcorpus.json', 'r') as settings_fp:
        settings = load_json(settings_fp, cls=None, object_hook=None, object_pairs_hook=None, parse_constant=None,
                             parse_float=None, parse_int=None, )
        logger.info('settings: {}'.format(settings))
        context_limit_ = settings['context_limit'] if 'context_limit' in settings.keys() else 100
        if 'context_limit' not in settings.keys():
            logger.warning('context limit not in settings; using default value {}.'.format(context_limit_))
        configuration['chainer']['pipe'][0]['context_limit'] = context_limit_
        logger.info('SQuAD context limit: {}'.format(configuration['chainer']['pipe'][0]['context_limit']))
        doc2vec_epochs = settings['doc2vec_epochs'] if 'doc2vec_epochs' in settings.keys() else 20
        if 'doc2vec_epochs' not in settings.keys():
            logger.warning('doc2vec epochs not in settings; using default value {}.'.format(doc2vec_epochs))
        else:
            logger.info('doc2vec epochs: {}'.format(doc2vec_epochs))

    with open(input_file, 'r') as input_fp:
        text = input_fp.read()
        text = text.split('\n')
        text = text[text_start:text_stop]
        text = ' '.join(text)
        logger.info('text length: {}'.format(len(text)))
        sentences = split_sentences(text)
        if pieces_strategy == pieces_strategies[0]:
            pieces = [text[i:i + context_limit_] for i in range(0, len(text), context_limit_)] + [
                text[i + context_limit_ // 2: i + 3 * context_limit_ // 2] for i in
                range(0, len(text) - context_limit_, context_limit_)]
        elif pieces_strategy == pieces_strategies[1]:
            pieces = [' '.join(sentences[index:index + sentences_per_chunk]) for index in
                      range(0, len(sentences), sentences_per_chunk)]
        else:
            raise ValueError(
                'pieces strategy can only be one of {} but is [{}]'.format(pieces_strategies, pieces_strategy))

        lower_pieces = [piece.lower() for piece in pieces]
        logger.info('context size: {} pieces: {}'.format(context_limit_, len(pieces)))
        if mode == modes[0]:
            vectorizer = TfidfVectorizer().fit(lower_pieces)
            pieces_ = vectorizer.transform(lower_pieces)
        elif mode == modes[1]:
            texts = [[word for word in tokenize_by_word(document.lower()) if word not in ENGLISH_STOP_WORDS] for
                     document in pieces]
            # remove words that appear only once
            frequency = Counter([token for text in texts for token in text])
            texts = [[token for token in text if frequency[token] > 1] for text in texts]
            dictionary = corpora.Dictionary(texts)
            logger.info('dictionary size: {}'.format(len(dictionary)))
            corpus_ = [dictionary.doc2bow(text) for text in texts]
            lsi = models.LsiModel(corpus_, id2word=dictionary, num_topics=lsi_topic_count)
            lsi.show_topics(num_topics=lsi_topic_count, num_words=100, log=True)
            matrix_similarity = MatrixSimilarity(lsi[corpus_], num_features=similarity_feature_count)
        elif mode in {modes[2], modes[3]}:
            texts = [[word for word in tokenize_by_word(document.lower()) if word not in ENGLISH_STOP_WORDS] for
                     document in pieces]
            # remove words that appear only once
            frequency = Counter([token for text in texts for token in text])
            texts = [[token for token in text if frequency[token] > 1] for text in texts]
            documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
            doc2vec_model = Doc2Vec(documents, epochs=doc2vec_epochs, min_count=doc2vec_min_count, seed=doc2vec_seed,
                                    vector_size=doc2vec_vector_size, window=doc2vec_window, workers=doc2vec_workers, )
            doc2vec_model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
            if mode == modes[2]:
                # pre-compute the vector for all of the pieces
                pieces_ = [doc2vec_model.infer_vector(piece.lower().split(), epochs=100) for piece in pieces]
        else:
            raise ValueError('mode can only be one of {} but is [{}]'.format(modes, mode))

    logger.info('building DeepPavlov model from configuration')
    model = build_model(configuration, download=True)

    logger.info('ready.')

    done = False
    while not done:
        question = input('?: ')
        # pre-process the question to remove trailing punctuation and trim space
        question = question.strip()
        while question.endswith('?'):
            question = question[:-1]
        question = question.strip()
        if question.lower() not in exit_questions:
            if mode == modes[0]:
                question_ = vectorizer.transform([question.lower()])
                cosine_similarities = cosine_similarity(question_, pieces_).flatten()
                related_product_indices = cosine_similarities.argsort()[:-results_to_return - 1:-1]
                cosine_format_ = 'Q: {} cos: {:5.3f} A: {}'
                if cosine_similarity(question_, pieces_[related_product_indices[0]])[0][0] != 0.0:
                    for index in related_product_indices:
                        current_ = cosine_similarity(question_, pieces_[index])[0][0]
                        result = model([pieces[index]], [question])
                        logger.info(cosine_format_.format(question, 1.0 - acos(current_) / pi, result[0]))
                else:
                    info(cosine_format_.format(question, 0.0, choice(miss_responses)))
            elif mode == modes[1]:
                question_ = lsi[dictionary.doc2bow(question.lower().split())]
                similarities = sorted(enumerate(matrix_similarity[question_]), key=lambda item: -item[1])[
                               :results_to_return]
                lsi_format_ = 'Q: {} : lsi: {} A: {}'
                if similarities[0][1] != 0.0:
                    for similarity in similarities:
                        result = model([pieces[similarity[0]]], [question.lower()])
                        info(lsi_format_.format(question, similarity, result[0]))
                else:
                    info(lsi_format_.format(question, 0.0, choice(miss_responses)))
            elif mode == modes[2]:
                question_ = doc2vec_model.infer_vector(question.lower().split(), epochs=doc2vec_question_epochs)
                similarities = sorted([(piece_index, 1.0 - acos(
                    cosine_similarity(question_.reshape(1, -1), piece_.reshape(1, -1), )) / pi) for
                                       piece_index, piece_
                                       in enumerate(pieces_)], key=lambda item: -item[1])[:results_to_return]
                d2vcos_format_ = 'Q: {} : d2v/cos: {:5.3f} A: {}'
                if similarities[0][1] != 0.0:
                    for similarity in similarities:
                        result = model([pieces[similarity[0]]], [question.lower()])
                        info(d2vcos_format_.format(question, similarity[1], result[0][0]))
                else:
                    info(d2vcos_format_.format(question, 0.0, choice(miss_responses)))
            elif mode == modes[3]:
                # https://stackoverflow.com/questions/42781292/doc2vec-get-most-similar-documents
                question_ = doc2vec_model.infer_vector(question.lower().split(), epochs=doc2vec_question_epochs)
                similarities = doc2vec_model.docvecs.most_similar([question_], topn=results_to_return)
                d2v_format_ = 'Q: {} : d2v: {:5.3f} A: {}'
                if similarities[0][1] != 0.0:
                    for similarity in similarities:
                        result = model([pieces[similarity[0]]], [question.lower()])
                        info(d2v_format_.format(question, similarity[1], result[0][0]))
                else:
                    info(d2v_format_.format(question, 0.0, choice(miss_responses)))
            else:
                raise ValueError('mode can only be one of {} but is [{}]'.format(modes, mode))
        else:
            done = True

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
