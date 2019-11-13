import logging
from time import time

from deeppavlov import train_model
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator


@register('my_dialog_iterator')
class MyDialogDatasetIterator(DataLearningIterator):

    @staticmethod
    def _dialogs(arg_data):
        dialogs = []
        prev_resp_act = None
        for x, y in arg_data:
            if x.get('episode_done'):
                del x['episode_done']
                prev_resp_act = None
                dialogs.append(([], []))
            x['prev_resp_act'] = prev_resp_act
            prev_resp_act = y['act']
            dialogs[-1][0].append(x)
            dialogs[-1][1].append(y)
        return dialogs

    def split(self, *args, **kwargs):
        self.train = self._dialogs(self.train)
        self.valid = self._dialogs(self.valid)
        self.test = self._dialogs(self.test)


network = {
    'api_call_action': 'api_call',
    'bow_embedder': {
        'class_name': 'bow',
        'depth': '#token_vocab.__len__()',
        'with_counts': True,
    },
    'class_name': 'go_bot',
    # this is the piece that is missing in the original
    'database': {
        'class_name': 'sqlite_database',
        'table_name': 'mytable',
        'primary_keys': ['name'],
        'save_path': '{DOWNLOADS_PATH}/dstc2/resto.sqlite'
    },
    'debug': False,
    'embedder': None,
    'in': ['x'],
    'intent_classifier': None,
    'in_y': ['y'],
    'load_path': '{MODELS_PATH}/my_gobot/model',
    'main': True,
    'network_parameters': {
        'dense_size': 160,
        'dropout_rate': 0.5,
        'hidden_size': 128,
        'l2_reg_coef': 7e-4,
        'learning_rate': 0.005,
    },
    'out': ['y_predicted'],
    'save_path': '{MODELS_PATH}/my_gobot/model',
    'slot_filler': {
        'config_path': '{DEEPPAVLOV_PATH}/configs/ner/slotfill_dstc2.json',
        # 'config_path': './slotfill.json',
    },
    'template_path': '{DOWNLOADS_PATH}/dstc2/dstc2-templates.txt',
    'template_type': 'DualTemplate',
    'tokenizer': {
        'class_name': 'stream_spacy_tokenizer',
        'lowercase': False
    },
    'tracker': {
        'class_name': 'featurized_tracker',
        'slot_names': ['pricerange', 'this', 'area', 'food', 'name']
    },
    'use_action_mask': False,
    'word_vocab': '#token_vocab',
}

tokenizer = {'class_name': 'deeppavlov.models.go_bot.wrapper:DialogComponentWrapper',
             'component': {'class_name': 'split_tokenizer'}, 'in': ['x'], 'out': ['x_tokens'], }

token_vocabulary = {'class_name': 'simple_vocab', 'fit_on': ['x_tokens'], 'id': 'token_vocab',
                    'load_path': '{MODELS_PATH}/my_gobot/token.dict',
                    'save_path': '{MODELS_PATH}/my_gobot/token.dict', }

basic_config = {
    'dataset_reader': {
        'class_name': 'dstc2_reader',
        'data_path': '{DOWNLOADS_PATH}/dstc2'
    },
    'dataset_iterator': {
        'class_name': 'my_dialog_iterator'
    },
    'chainer': {
        'in': ['x'],
        'in_y': ['y'],
        'out': ['y_predicted'],
        'pipe': [
            tokenizer,
            token_vocabulary,
            network
        ]
    },
    'train': {
        'epochs': 200,
        'batch_size': 4,
        'metrics': ['per_item_dialog_accuracy'],
        'validation_patience': 10,
        'val_every_n_batches': 15,
        'val_every_n_epochs': -1,
        'log_every_n_batches': 15,
        'log_every_n_epochs': -1,
        'show_examples': False,
        'validate_best': True,
        'test_best': True
    },
    'metadata': {
        'variables': {
            'ROOT_PATH': '~/.deeppavlov',
            'DOWNLOADS_PATH': '{ROOT_PATH}/downloads',
            'MODELS_PATH': './models',
            'CONFIGS_PATH': './configs'
        },
        'requirements': [
            '{DEEPPAVLOV_PATH}/requirements/tf.txt',
            '{DEEPPAVLOV_PATH}/requirements/fasttext.txt',
            '{DEEPPAVLOV_PATH}/requirements/spacy.txt',
            '{DEEPPAVLOV_PATH}/requirements/en_core_web_sm.txt'
        ],
        'labels': {
            'telegram_utils': 'GoalOrientedBot',
            'server_utils': 'GoalOrientedBot'
        },
        'download': [
            {
                'url': 'http://files.deeppavlov.ai/datasets/dstc2_v2.tar.gz',
                'subdir': '{DOWNLOADS_PATH}/dstc2'
            }
        ]
    }
}

if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    basic_bot = train_model(basic_config, download=True)

    for question in ['hello', 'I want some chinese food', 'on the south side?',
                     'i want cheap food in chinese restaurant in the south of town',
                     'bye']:
        answer = basic_bot([question])
        logger.info('Q: {} A: {}'.format(question, answer))

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
