import logging
from time import time

from deeppavlov import build_model
from deeppavlov import configs
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


if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info('started')

    data = DSTC2DatasetReader().read('./dstc2')
    iterator = MyDialogDatasetIterator(data)

    x_dialog, y_dialog = iterator.train[0]
    logger.info('x size: {} y size: {}'.format(len(x_dialog), len(y_dialog)))
    logger.info('data keys: {}'.format(list(data.keys())))
    for item in data['train'][:10]:
        logger.info(item)
        for key, value in item[0].items():
            logger.info('{} :: {}'.format(key, value))

    gobot_config = configs.go_bot.gobot_dstc2
    bot = build_model(gobot_config, download=True)
    for question in ['Hello.', 'Hi, I want some food.', 'I would like Indian food instead.', ]:
        answer = bot([question])
        logger.info('Q: {} A: {}'.format(question, answer))
    bot.reset()

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
