import logging
from time import time

from deeppavlov import build_model

context_limit_ = 400
configuration = {
    'chainer': {
        'in': ['context_raw', 'question_raw'],
        'in_y': ['ans_raw', 'ans_raw_start'],
        'pipe': [{'char_limit': 16,
                  'class_name': 'squad_preprocessor',
                  'context_limit': context_limit_,  # was 400
                  'id': 'squad_prepr',
                  'in': ['context_raw', 'question_raw'],
                  'out': ['context', 'context_tokens', 'context_chars',
                          'c_r2p', 'c_p2r', 'question',
                          'question_tokens', 'question_chars', 'spans'],
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
        'requirements': ['{DEEPPAVLOV_PATH}/requirements/tf.txt', ],
        'labels': {'server_utils': 'SquadModel', 'telegram_utils': 'SquadModel', },
        'download': [{'subdir': '{MODELS_PATH}',
                      'url': 'http://files.deeppavlov.ai/deeppavlov_data/squad_model_1.4_cpu_compatible.tar.gz', },
                     {'subdir': '{DOWNLOADS_PATH}/embeddings',
                      'url': 'http://files.deeppavlov.ai/embeddings/wiki-news-300d-1M.vec', },
                     {'subdir': '{DOWNLOADS_PATH}/embeddings',
                      'url': 'http://files.deeppavlov.ai/embeddings/wiki-news-300d-1M-char.vec', }],
    },
    'train': {'batch_size': 50, 'class_name': 'nn_trainer', 'evaluation_targets': ['valid'], 'log_every_n_batches': 250,
              'metrics': [{'inputs': ['ans_raw', 'ans_predicted'], 'name': 'squad_v1_em', },
                          {'inputs': ['ans_raw', 'ans_predicted'], 'name': 'squad_v1_f1', }, ], 'pytest_max_batches': 2,
              'show_examples': False, 'val_every_n_epochs': 1, 'validation_patience': 10, },
}
input_file = './data/35830.txt'
if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info('started')

    model = build_model(configuration, download=True)
    with open(input_file, 'r') as input_fp:
        text = input_fp.read()
        text = text.split('\n')
        text = text[2124:524200]
        text = ' '.join(text)
        logger.info('text length: {}'.format(len(text)))
    logger.info('ready.')

    done = False
    while not done:
        question = input('?: ')
        if question not in {'bye', 'quit'}:
            logger.info('Q: {} A: {}'.format(question, model([text], [question])))
        else:
            done = True

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
