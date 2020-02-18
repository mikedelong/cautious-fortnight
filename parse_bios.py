from glob import glob
from json import load as json_load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started.')

    with open('./parse_bios.json', 'r') as settings_fp:
        settings = json_load(settings_fp, cls=None, object_hook=None, parse_float=None, parse_int=None,
                             parse_constant=None, object_pairs_hook=None)

    input_folder = settings['input_folder'] if 'input_folder' in settings.keys() else None
    if input_folder:
        logger.info('input folder: {}'.format(input_folder))
    else:
        logger.warning('input folder is None. Quitting.')
        quit(code=1)

    input_files = [input_file for input_file in glob(input_folder + '*')]
    for input_file in input_files:
        input_file = input_file.replace('\\', '/')
        with open(input_file, 'r') as input_fp:
            content = input_fp.read()
            logger.info('file: {} size: {}'.format(input_file, len(content)))

    logger.info('input file count: {}'.format(len(input_files)))
