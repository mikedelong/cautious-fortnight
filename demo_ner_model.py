import logging
from time import time

from deeppavlov import build_model
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


dslotfill = {
    'dataset_reader': {'class_name': 'dstc2_reader', 'data_path': '{DATA_PATH}', },
    'dataset_iterator': {'class_name': 'dstc2_ner_iterator', 'slot_values_path': '{SLOT_VALS_PATH}'},
    'chainer': {'in': ['x'], 'in_y': ['y'], 'out': ['slots'],
                'pipe': [{'class_name': 'lazy_tokenizer', 'in': ['x'], 'out': ['x_tokens'], },
                         {'config_path': '{NER_CONFIG_PATH}', 'in': ['x_tokens'], 'out': ['x_tokens', 'tags'], },
                         {'class_name': 'dstc_slotfilling', 'in': ['x_tokens', 'tags'],
                          'load_path': '{MODEL_PATH}/model',
                          'out': ['slots'], 'save_path': '{MODEL_PATH}/model', 'threshold': 0.8, }], },
    'train': {'class_name': 'fit_trainer', 'evaluation_targets': ['valid', 'test'], 'metrics': ['slots_accuracy'], },
    'metadata': {
        'variables': {
            'ROOT_PATH': '~/.deeppavlov',
            # 'NER_CONFIG_PATH': '{DEEPPAVLOV_PATH}/configs/ner/ner_dstc2.json',
            'NER_CONFIG_PATH': './ner_dstc2.json',
            'DATA_PATH': '{ROOT_PATH}/downloads/dstc2',
            'SLOT_VALS_PATH': '{DATA_PATH}/dstc_slot_vals.json',
            'MODELS_PATH': '{ROOT_PATH}/models',
            'MODEL_PATH': '{MODELS_PATH}/slotfill_dstc2',
        },
        # 'requirements': ['{DEEPPAVLOV_PATH}/requirements/tf.txt'],
        'requirements': ['./tf.txt'],
        'labels': {'telegram_utils': 'NERModel', 'server_utils': 'DstcSlotFillingNetwork'},
        'download': [
            {'subdir': '{DATA_PATH}', 'url': 'http://files.deeppavlov.ai/deeppavlov_data/dstc_slot_vals.tar.gz', },
            {'subdir': '{MODELS_PATH}', 'url': 'http://files.deeppavlov.ai/deeppavlov_data/slotfill_dstc2.tar.gz', }
        ]
    }
}

if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info('started')

    ner_model = build_model(dslotfill, download=True)
    for question in ['i want cheap food in chinese restaurant in the south of town',
                     'I want cheap Chinese food on the south side of town.']:
        slots = ner_model([question])[0]
        logger.info('Q: {} S: {}'.format(question, slots))

    logger.info(dslotfill)
    logger.info('total time: {:5.2f}s'.format(time() - time_start))
