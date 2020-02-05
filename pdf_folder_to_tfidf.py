from glob import glob
from json import load as json_load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from string import punctuation
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from tika import parser
from unidecode import unidecode

PUNCTUATION = set(punctuation)


def ispunct(arg):
    for character in arg:
        if character not in PUNCTUATION:
            return False
    return True


def pdf_to_text(arg):
    parse_result = parser.from_file(arg)
    if parse_result['content']:
        result = unidecode(parse_result['content'])
        if len(result):
            return result
        else:
            print('\'{}\','.format(arg))
    else:
        print('\'{}\','.format(arg))
        return ' '


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

    # todo split PDF parsing and TFIDF into separate loops

    vectorizer = TfidfVectorizer(
        # input='files',
        # input='content',
        input='filename',
        encoding='utf-8',
        decode_error='strict', strip_accents=None, lowercase=True,
        preprocessor=pdf_to_text,
        # preprocessor=None,
        tokenizer=None, analyzer='word', stop_words=None, token_pattern='(?u)\b\w\w+\b',
        ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None,
        binary=False,
        # dtype=<class 'numpy.float64'>,
        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)

    input_files = [input_file.replace('\\', '/') for input_file in glob(input_folder + '*.pdf')
                   if input_file.replace('\\', '/') not in {
                   }]
    # t = [pdf_to_text(item) for item in input_files[:2]]
    model = vectorizer.fit_transform(input_files)
    logger.info('model shape: {}'.format(len(model)))
