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
