from deeppavlov import build_model
from deeppavlov import configs
from deeppavlov import train_model
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader


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


dslotfill = {
    'dataset_reader': {
        'class_name': 'dstc2_reader',
        'data_path': '{DATA_PATH}'
    },
    'dataset_iterator': {
        'class_name': 'dstc2_ner_iterator',
        'slot_values_path': '{SLOT_VALS_PATH}'
    },
    'chainer': {
        'in': ['x'],
        'in_y': ['y'],
        'pipe': [
            {
                'in': ['x'],
                'class_name': 'lazy_tokenizer',
                'out': ['x_tokens']
            },
            {
                'in': ['x_tokens'],
                'config_path': '{NER_CONFIG_PATH}',
                'out': ['x_tokens', 'tags']
            },

            {
                'in': ['x_tokens', 'tags'],
                'class_name': 'dstc_slotfilling',
                'threshold': 0.8,
                'save_path': '{MODEL_PATH}/model',
                'load_path': '{MODEL_PATH}/model',
                'out': ['slots']
            }
        ],
        'out': ['slots']
    },
    'train': {
        'metrics': ['slots_accuracy'],
        'class_name': 'fit_trainer',
        'evaluation_targets': [
            'valid',
            'test'
        ]
    },
    'metadata': {
        'variables': {
            'ROOT_PATH': '~/.deeppavlov',
            'NER_CONFIG_PATH': '{DEEPPAVLOV_PATH}/configs/ner/ner_dstc2.json',
            'DATA_PATH': '{ROOT_PATH}/downloads/dstc2',
            'SLOT_VALS_PATH': '{DATA_PATH}/dstc_slot_vals.json',
            'MODELS_PATH': '{ROOT_PATH}/models',
            'MODEL_PATH': '{MODELS_PATH}/slotfill_dstc2'
        },
        'requirements': [
            '{DEEPPAVLOV_PATH}/requirements/tf.txt'
        ],
        'labels': {
            'telegram_utils': 'NERModel',
            'server_utils': 'DstcSlotFillingNetwork'
        },
        'download': [
            {
                'url': 'http://files.deeppavlov.ai/deeppavlov_data/dstc_slot_vals.tar.gz',
                'subdir': '{DATA_PATH}'
            },
            {
                'url': 'http://files.deeppavlov.ai/deeppavlov_data/slotfill_dstc2.tar.gz',
                'subdir': '{MODELS_PATH}'
            }
        ]
    }
}

network = {
    'api_call_action': 'api_call',
    'bow_embedder': {
        'class_name': 'bow',
        'depth': '#token_vocab.__len__()',
        'with_counts': True,
    },
    'class_name': 'go_bot',
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
    data = DSTC2DatasetReader().read('./dstc2')
    iterator = MyDialogDatasetIterator(data)

    x_dialog, y_dialog = iterator.train[0]
    print('x size: {} y size: {}'.format(len(x_dialog), len(y_dialog)))
    bot = build_model(configs.go_bot.gobot_dstc2, download=True)
    for question in ['hi i want some food', 'i would like indian food instead', ]:
        answer = bot([question])
        print('Q: {} A: {}'.format(question, answer))
    bot.reset()

    ner_model = build_model(dslotfill, download=True)
    for question in ['i want cheap food in chinese restaurant in the south of town']:
        slots = ner_model([question])[0]
        print('Q: {} S: {}'.format(question, slots))

    basic_bot = train_model(basic_config, download=True)

    for question in ['hello', 'I want some chinese food', 'on the south side?', 'maybe indian?', 'bye']:
        answer = basic_bot([question])
        print('Q: {} A: {}'.format(question, answer))
