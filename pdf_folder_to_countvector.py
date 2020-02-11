from glob import glob
from json import dump as json_dump
from json import load as json_load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from string import punctuation
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from tika import parser
from unidecode import unidecode

PUNCTUATION = set(punctuation)


def ispunct(arg):
    for character in arg:
        if character not in PUNCTUATION:
            return False
    return True


if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started.')

    with open('./pdf_folder_to_counts.json', 'r') as settings_fp:
        settings = json_load(settings_fp, cls=None, object_hook=None, parse_float=None, parse_int=None,
                             parse_constant=None, object_pairs_hook=None)

    filter_threshold = settings['filter_threshold'] if 'filter_threshold' in settings.keys() else 1
    if 'filter_threshold' in settings.keys():
        logger.info('filter threshold: {}'.format(filter_threshold))
    else:
        logger.warning('filter threshold not in settings; using default {}'.format(filter_threshold))
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
    verbs = settings['verbs'] if 'verbs' in settings.keys() else dict()
    if len(verbs):
        with open(settings['verbs'], 'r') as verbs_fp:
            verbs = json_load(verbs_fp)
            logger.info('verbs: {}'.format(verbs))
    else:
        logger.warning('verbs not in settings; we will not be doing any verbal-form consolidation')

    # do a sanity check on our plurals and verbs
    collisions = set(plurals.keys()).intersection(set(verbs.keys()))
    if len(collisions):
        logger.warning('we have plural/verb collisions: {}. Qutting.'.format(collisions))
        quit(code=3)

    input_files = [input_file for input_file in glob(input_folder + '*.pdf')]
    items = list()
    for input_file in input_files:
        parse_result = parser.from_file(input_file)
        if parse_result['content']:
            items.append(unidecode(parse_result['content']))
            logger.info('length: {} name: {}'.format(len(parse_result['content']), input_file))
        else:
            logger.warning('length: 0 name: {}'.format(input_file))

    logger.info('result size: {}'.format(len(items)))
    logger.info('file count: {}'.format(len(input_files)))

    # add a map of singulars to plurals to complement our plurals to singulars map
    singulars = {plurals[key]: key for key in plurals.keys()}

    # todo: factor these out as data
    capitalization = {'AFFAIRS', 'AID', 'AMBASSADOR', 'Attendees', 'Code', 'Coordination', 'File', 'Location',
                      'Meeting', 'Please',
                      'Project', 'Prepared', 'Purpose', 'RECORD', 'Record', 'Recording', 'Research', 'SUBJECT',
                      'Thanks', 'Title', 'Topics', 'Yes', }
    split = {'AFGHAN': ['Afghan'], 'AFGHANISTAN': ['Afghanistan'], 'AMERICA': ['America'],
             'AMERICA1:1': ['America'],
             'ofthe': ['of', 'the'], 'Date/Time': ['date', 'time'], '(Name,title': ['name', 'title'],
             'Recordof': ['record', 'of'], 'U.S.': ['US'], 'wantto': ['want', 'to'], 'wasa': ['was', 'a'], }

    text = ''
    for item_index, item in enumerate(items):
        if item is not None:
            pieces = [piece.strip() for piece in item.split()]
            pieces = [[piece] if piece not in split.keys() else split[piece] for piece in pieces]
            pieces = [item for piece in pieces for item in piece]
            pieces = [piece[1:] if piece.startswith('(') and ')(' not in piece else piece for piece in pieces]
            pieces = [piece[:-1] if piece.endswith(')') and ')(' not in piece else piece for piece in pieces]
            for punctuation in ['\'', '\"', '[', ]:
                pieces = [piece if not piece.startswith(punctuation) else piece[1:] for piece in pieces]
            for punctuation in [':', ';', '.', ',', '?', '\'', '\"', ']', '|', '/']:
                pieces = [piece if not piece.endswith(punctuation) else piece[:-1] for piece in pieces]
            pieces = [piece for piece in pieces if len(piece) > 1]
            pieces = [piece for piece in pieces if not piece.isdigit()]
            pieces = [piece if piece not in plurals.keys() else '{}/{}'.format(plurals[piece], piece)
                      for piece in pieces]
            pieces = [piece if piece not in singulars.keys() else '{}/{}'.format(piece, singulars[piece])
                      for piece in pieces]
            pieces = [piece if piece not in capitalization else piece.lower() for piece in pieces]
            pieces = [piece if piece not in verbs.keys() else '{}'.format(verbs[piece]) for piece in pieces]
            pieces = [piece for piece in pieces if not ispunct(piece)]
            text += ' '.join(pieces)

    # todo tune this so it captures 2-grams
    vectorizer = CountVectorizer()
    fit_result = vectorizer.fit_transform([text])
    result = dict(zip(vectorizer.get_feature_names(), fit_result.toarray().sum(axis=0)))
    with open(output_file, 'w') as output_fp:
        json_dump(result, output_fp, sort_keys=True)
