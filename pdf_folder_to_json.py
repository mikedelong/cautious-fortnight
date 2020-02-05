from glob import glob
from json import dump as json_dump
from json import load as json_load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

from tika import parser
from unidecode import unidecode

if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started.')

    with open('./pdf_folder_to_json.json', 'r') as settings_fp:
        settings = json_load(settings_fp, cls=None, object_hook=None, parse_float=None, parse_int=None,
                             parse_constant=None, object_pairs_hook=None)

    input_folder = settings['input_folder'] if 'input_folder' in settings.keys() else None
    if input_folder:
        logger.info('input folder: {}'.format(input_folder))
    else:
        logger.warning('input folder is None. Quitting.')
        quit(code=1)
    output_file = settings['output_file'] if 'output_file' in settings.keys() else None
    if output_file:
        logger.info('output file: {}'.format(output_file))
    else:
        logger.warning('output file is missing from the settings. Quitting.')
        quit(code=2)

    items = dict()
    input_files = [input_file for input_file in glob(input_folder + '*.pdf')]
    for input_file in input_files:
        parse_result = parser.from_file(input_file)
        if parse_result['content']:
            key = input_file.replace('//', '/')
            value = unidecode(parse_result['content'])
            if len(value) > 0:
                items[key] = value
                logger.info('size: {} name: {}'.format(len(value), key))
            else:
                logger.warning('{} content is empty.'.format(key))

    logger.info('result size: {}'.format(len(items)))
    logger.info('file count: {}'.format(len(input_files)))

    with open(output_file, 'w') as output_fp:
        json_dump(items, output_fp, sort_keys=True)
