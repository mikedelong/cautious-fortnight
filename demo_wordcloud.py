from glob import glob
from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

input_folder = './afghanistan-papers-documents/'
if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started')

    input_file_count = 0
    for input_file_index, input_file in enumerate(glob(input_folder + '*.pdf')):
        logger.info(input_file)
        input_file_count += 1

    logger.info('file count: {}'.format(input_file_count))

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
