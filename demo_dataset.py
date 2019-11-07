from deeppavlov import build_model
from deeppavlov import configs
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader


@register('my_dialog_iterator')
class MyDialogDatasetIterator(DataLearningIterator):

    @staticmethod
    def _dialogs(data):
        dialogs = []
        prev_resp_act = None
        for x, y in data:
            if x.get('episode_done'):
                del x['episode_done']
                prev_resp_act = None
                dialogs.append(([], []))
            x['prev_resp_act'] = prev_resp_act
            prev_resp_act = y['act']
            dialogs[-1][0].append(x)
            dialogs[-1][1].append(y)
        return dialogs

    # @overrides
    def split(self, *args, **kwargs):
        self.train = self._dialogs(self.train)
        self.valid = self._dialogs(self.valid)
        self.test = self._dialogs(self.test)


dslotfill = {
    "dataset_reader": {
        "class_name": "dstc2_reader",
        "data_path": "{DATA_PATH}"
    },
    "dataset_iterator": {
        "class_name": "dstc2_ner_iterator",
        "slot_values_path": "{SLOT_VALS_PATH}"
    },
    "chainer": {
        "in": ["x"],
        "in_y": ["y"],
        "pipe": [
            {
                "in": ["x"],
                "class_name": "lazy_tokenizer",
                "out": ["x_tokens"]
            },
            {
                "in": ["x_tokens"],
                "config_path": "{NER_CONFIG_PATH}",
                "out": ["x_tokens", "tags"]
            },

            {
                "in": ["x_tokens", "tags"],
                "class_name": "dstc_slotfilling",
                "threshold": 0.8,
                "save_path": "{MODEL_PATH}/model",
                "load_path": "{MODEL_PATH}/model",
                "out": ["slots"]
            }
        ],
        "out": ["slots"]
    },
    "train": {
        "metrics": ["slots_accuracy"],
        "class_name": "fit_trainer",
        "evaluation_targets": [
            "valid",
            "test"
        ]
    },
    "metadata": {
        "variables": {
            "ROOT_PATH": "~/.deeppavlov",
            "NER_CONFIG_PATH": "{DEEPPAVLOV_PATH}/configs/ner/ner_dstc2.json",
            "DATA_PATH": "{ROOT_PATH}/downloads/dstc2",
            "SLOT_VALS_PATH": "{DATA_PATH}/dstc_slot_vals.json",
            "MODELS_PATH": "{ROOT_PATH}/models",
            "MODEL_PATH": "{MODELS_PATH}/slotfill_dstc2"
        },
        "requirements": [
            "{DEEPPAVLOV_PATH}/requirements/tf.txt"
        ],
        "labels": {
            "telegram_utils": "NERModel",
            "server_utils": "DstcSlotFillingNetwork"
        },
        "download": [
            {
                "url": "http://files.deeppavlov.ai/deeppavlov_data/dstc_slot_vals.tar.gz",
                "subdir": "{DATA_PATH}"
            },
            {
                "url": "http://files.deeppavlov.ai/deeppavlov_data/slotfill_dstc2.tar.gz",
                "subdir": "{MODELS_PATH}"
            }
        ]
    }
}

network = {
    "in": ["x"],
    "in_y": ["y"],
    "out": ["y_predicted"],
    "main": True,
    "class_name": "go_bot",
    "load_path": "{MODELS_PATH}/my_gobot/model",
    "save_path": "{MODELS_PATH}/my_gobot/model",
    "debug": False,
    "word_vocab": "#token_vocab",
    "template_path": "{DOWNLOADS_PATH}/dstc2/dstc2-templates.txt",
    "template_type": "DualTemplate",
    "api_call_action": "api_call",
    "use_action_mask": False,
    "network_parameters": {
        "learning_rate": 0.005,
        "dropout_rate": 0.5,
        "l2_reg_coef": 7e-4,
        "hidden_size": 128,
        "dense_size": 160
    },
    "slot_filler": None,
    "intent_classifier": None,
    "embedder": None,
    "bow_embedder": {
        "class_name": "bow",
        "depth": "#token_vocab.__len__()",
        "with_counts": True,
        # },
        # "slot_filler": {
        "config_path": "{DEEPPAVLOV_PATH}/configs/ner/slotfill_dstc2.json"
    },
    "tokenizer": {
        "class_name": "stream_spacy_tokenizer",
        "lowercase": False
    },
    "tracker": {
        "class_name": "featurized_tracker",
        "slot_names": ["pricerange", "this", "area", "food", "name"]
    }
}

tokenizer = {'class_name': 'deeppavlov.models.go_bot.wrapper:DialogComponentWrapper',
             'component': {'class_name': 'split_tokenizer'}, 'in': ['x'], 'out': ['x_tokens'], }

token_vocabulary = {'class_name': 'simple_vocab', 'fit_on': ['x_tokens'], 'id': 'token_vocab',
                    'load_path': '{MODELS_PATH}/my_gobot/token.dict',
                    'save_path': '{MODELS_PATH}/my_gobot/token.dict', }

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
    for question in ['i want cheap food in chinese reastaurant in the south of town']:
        slots = ner_model([question])[0]
        print('Q: {} S: {}'.format(question, slots))
