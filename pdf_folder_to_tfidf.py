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

    vectorizer = TfidfVectorizer(
        # input='files',
        input='content',
        encoding='utf-8',
        decode_error='strict', strip_accents=None, lowercase=True, preprocessor=pdf_to_text,
        tokenizer=None, analyzer='word', stop_words=None, token_pattern='(?u)\b\w\w+\b',
        ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None,
        binary=False,
        # dtype=<class 'numpy.float64'>,
        norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)

    input_files = [input_file for input_file in glob(input_folder + '*.pdf') if input_file.replace('\\', '/') not in {
        './afghanistan-papers-documents/background_ll_01_xx2_dc_07102015.pdf',
        './afghanistan-papers-documents/background_ll_01_xx_dc_05052015.pdf',
        './afghanistan-papers-documents/background_ll_01_xx_dc_08252015.pdf',
        './afghanistan-papers-documents/background_ll_01_xx_nyc_01202015.pdf',
        './afghanistan-papers-documents/background_ll_01_xx_phone_08042015.pdf',
        './afghanistan-papers-documents/background_ll_01_xx_phone_08252014.pdf',
        './afghanistan-papers-documents/background_ll_01_xx_xx2_06252015.pdf',
        './afghanistan-papers-documents/background_ll_01_xx_xx_04092015.pdf',
        './afghanistan-papers-documents/background_ll_01_xx_xx_08202015.pdf',
        './afghanistan-papers-documents/background_ll_02_xx_mainstate_06092015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx2_dc_12162015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_dc_04072016.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_dc_09112015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_dc_12082015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_kabul2_10202015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_kabul_10202015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_phone_03012016.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_xx2_08242015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_xx_07272015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_xx_07302015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_xx_08122015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_xx_08242015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_xx_08262015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_xx_08272015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_xx_08282015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_xx_09012015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_xx_09242015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_xx_10012015.pdf',
        './afghanistan-papers-documents/background_ll_03_xx_xx_10212015.pdf',
        './afghanistan-papers-documents/background_ll_04_xx2_08312016.pdf',
        './afghanistan-papers-documents/background_ll_04_xx_05102016.pdf',
        './afghanistan-papers-documents/background_ll_05_xx_xx_12152015.pdf',
        './afghanistan-papers-documents/background_ll_07_xx_skype_07132016_17.pdf',
        './afghanistan-papers-documents/background_ll_07_xx_woodbridge_08032016.pdf',
        './afghanistan-papers-documents/background_ll_civilianroundtable_12042014.pdf',
        './afghanistan-papers-documents/background_ll_interiorministry_emails_04052016.pdf',
        './afghanistan-papers-documents/background_ll_literacy_emails_04042017.pdf',
        './afghanistan-papers-documents/background_ll_militaryroundtable_01142015.pdf',
        './afghanistan-papers-documents/cahill_dennis_ll_07_xx_dc_03082017.pdf',
        './afghanistan-papers-documents/chretien_marc_ll_07_xx_arlington_07222016.pdf',
        './afghanistan-papers-documents/johnson_thomas_ll_01072016.pdf',
        './afghanistan-papers-documents/khalilzad_zalma_ll_12072016.pdf',
        './afghanistan-papers-documents/williams_mike_ll_07_74_02272018.pdf',
        './afghanistan-papers-documents/zia_ehsan_ll001302017.pdf',
        './afghanistan-papers-documents/yamashita_ken_ll_05_a7_12152015.pdf',
        './afghanistan-papers-documents/yamashita_ken_ll_03_51_03292016.pdf',
    }]
    # t = [pdf_to_text(item) for item in input_files[:2]]
    model = vectorizer.fit_transform(input_files)
    logger.info('model shape: {}'.format(len(model)))
