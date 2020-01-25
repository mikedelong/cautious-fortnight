from collections import Counter
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

    with open('./pdf_folder_to_counts.json', 'r') as settings_fp:
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
    plurals = settings['plurals'] if 'plurals' in settings.keys() else dict()
    if len(plurals):
        with open(settings['plurals'], 'r') as plurals_fp:
            plurals = json_load(plurals_fp)
            logger.info('plurals: {}'.format(plurals))
    else:
        logger.warning('plurals not in settings; we will not be doing any singular/plural reconciliation')
    stop_word = settings['stop_word'] if 'stop_word' in settings.keys() else list()
    if len(stop_word):
        logger.info('stop word list: {}'.format(stop_word))
    else:
        logger.warning('stop word list not in settings; default is empty.')
    token_count = settings['token_count'] if 'token_count' in settings.keys() else 10
    if 'token_count' in settings.keys():
        logger.info('token count: {}'.format(token_count))
    else:
        logger.warning('token count not in settings; default value is {}.'.format(token_count))

    input_files = [input_file for input_file in glob(input_folder + '*.pdf')]
    logger.info('file count: {}'.format(len(input_files)))
    items = list()
    for input_file_index, input_file in enumerate(glob(input_folder + '*.pdf')):
        logger.info(input_file)
        parse_result = parser.from_file(input_file)
        if parse_result['content']:
            items.append(unidecode(parse_result['content']))

    logger.info('result size: {}'.format(len(items)))

    # add a map of singulars to plurals to complement our plurals to singulars map
    singulars = {plurals[key]: key for key in plurals.keys()}

    # first get all the counts
    count = Counter()
    for item_index, item in enumerate(items):
        if item is not None:
            pieces = [piece.strip() for piece in item.split()]
            pieces = [piece if piece not in {'U.S.'} else 'US' for piece in pieces]
            pieces = [piece[1:] if piece.startswith('(') and ')(' not in piece else piece for piece in pieces]
            pieces = [piece[:-1] if piece.endswith(')') and ')(' not in piece else piece for piece in pieces]
            for punctuation in ['\'', '\"', '[', ]:
                pieces = [piece if not piece.startswith(punctuation) else piece[1:] for piece in pieces]
            for punctuation in [':', ';', '.', ',', '?', '\'', '\"', ']', ]:
                pieces = [piece if not piece.endswith(punctuation) else piece[:-1] for piece in pieces]
            pieces = [piece if piece not in plurals.keys() else '{}/{}'.format(piece, plurals[piece])
                      for piece in pieces]
            pieces = [piece if piece not in singulars.keys() else '{}/{}'.format(singulars[piece], piece)
                      for piece in pieces]
            pieces = [piece if piece not in {'AID', 'AMBASSADOR', 'Meeting', 'Please', 'RECORD', 'Record', 'SUBJECT',
                                             'Title', 'Yes', } else piece.lower() for piece in pieces]
            for piece in pieces:
                count[piece] += 1 if all([len(piece) > 1, not piece.isdigit(), ]) else 0

    # filter out all the tokens that appear only once
    count = Counter({item: count[item] for item in count if count[item] > 1})
    logger.info('stop words: {}'.format(sorted(stop_word)))

    with open(output_file, 'w') as output_fp:
        json_dump(dict(count), output_fp, sort_keys=True)
