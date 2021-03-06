import logging
from time import time

from deeppavlov import build_model

configuration = {
    'chainer': {
        'in': ['context_raw', 'question_raw'],
        'in_y': ['ans_raw', 'ans_raw_start'],
        'pipe': [{'char_limit': 16,
                  'class_name': 'squad_preprocessor',
                  'context_limit': 400,
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
if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info('started')

    model = build_model(configuration, download=True)
    data = {
        'Apollo 11 was the spaceflight that first landed humans on the Moon. Commander Neil Armstrong and '
        'lunar module pilot Buzz Aldrin formed the American crew that landed the Apollo Lunar Module Eagle on '
        'July 20, 1969, at 20:17 UTC. Armstrong became the first person to step onto the lunar surface six hours '
        'and 39 minutes later on July 21 at 02:56 UTC; Aldrin joined him 19 minutes later. They spent about '
        'two and a quarter hours together outside the spacecraft, and they collected 47.5 pounds (21.5 kg) of '
        'lunar material to bring back to Earth. Command module pilot Michael Collins flew the command module '
        'Columbia alone in lunar orbit while they were on the Moon\'s surface. Armstrong and Aldrin spent 21 hours, '
        '36 minutes on the lunar surface at a site they named Tranquility Base before lifting off to rejoin '
        'Columbia in lunar orbit.': [
            'Has anyone been to the moon?',
            'Who comprised the Apollo 11 crew?',
            'When was the first moon landing?',
            'When did Eagle land?',
            'What was the Apollo 11 landing site named?',
            'What was the landing site named?',
            'What was the site named?',
            'Who was the Eagle pilot?',
            'Who was the lunar module pilot?',
            'Who flew the command module?',
            'What was the name of the command module?',
            'How much lunar material did Apollo 11 bring back?',
            'How long were Armstrong and Aldrin on the moon?',
            'How long were Armstrong and Aldrin on the lunar surface?',
        ],
        'No. 33 Squadron is a Royal Australian Air Force (RAAF) strategic transport and air-to-air refuelling '
        'squadron. It operates Airbus KC-30A Multi Role Tanker Transports from RAAF Base Amberley, Queensland. '
        'The squadron was formed in February 1942 for service during World War II, operating Short Empire flying '
        'boats and a variety of smaller aircraft. By 1944 it had completely re-equipped with Douglas C-47 Dakota '
        'transports, which it flew in New Guinea prior to disbanding in May 1946. The unit was re-established in '
        'February 1981 as a flight, equipped with two Boeing 707s for VIP and other long-range transport duties out '
        'of RAAF Base Richmond, New South Wales. No. 33 Flight was re-formed as a full squadron in July 1983. By 1988 '
        'it was operating six 707s, four of which were subsequently converted for aerial refuelling. The 707s saw '
        'active service during operations in Namibia, Somalia, the Persian Gulf, and Afghanistan. One of the '
        'transport jets was lost in a crash in October 1991. No. 33 Squadron relocated to Amberley and was '
        'temporarily without aircraft following the retirement of the 707s in June 2008. It began re-equipping '
        'with KC-30As in June 2011, and achieved initial operating capability with the type in February 2013. One of '
        'its aircraft was deployed to the Middle East in September 2014, as part of Australia\'s contribution to the '
        'military coalition against ISIS.': [
            'When was Number 33 Squadron re-established?',
            'How many planes were in Number 33 Squadron?',
            'What kind of planes were in Number 33 Squadron?',
            'Number 33 Squadron was equipped with what kind of planes?',
            'Number 33 Squadron operated what kind of planes?',
            'Number 33 Squadron operates what kind of planes?',
            'When did the squadron lose a plane in a crash?',
            'When did the squadron retire its 707s?',
        ],
    }
    for text, questions in data.items():
        for question in questions:
            logger.info('Q: {} A: {}'.format(question, model([text], [question])))

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
