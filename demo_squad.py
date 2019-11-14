import logging
from time import time

from deeppavlov import build_model
from deeppavlov import configs

if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info('started')

    model = build_model(configs.squad.squad, download=True)
    result = model(['DeepPavlov is library for NLP and dialog systems.'], ['What is DeepPavlov?'])
    logger.info(result)

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
