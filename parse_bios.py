from glob import glob
from json import load as json_load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

from bs4 import BeautifulSoup

if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started.')

    with open('./parse_bios.json', 'r') as settings_fp:
        settings = json_load(settings_fp, cls=None, object_hook=None, parse_float=None, parse_int=None,
                             parse_constant=None, object_pairs_hook=None)

    file_pattern = settings['file_pattern'] if 'file_pattern' in settings.keys() else None
    if file_pattern:
        logger.info('file pattern: {}'.format(file_pattern))
    else:
        logger.warning('file pattern is None. Quitting.')
        quit(code=1)

    input_folder = settings['input_folder'] if 'input_folder' in settings.keys() else None
    if input_folder:
        logger.info('input folder: {}'.format(input_folder))
    else:
        logger.warning('input folder is None. Quitting.')
        quit(code=2)

    input_files = [input_file for input_file in glob(input_folder + file_pattern)]
    for input_file in input_files:
        input_file = input_file.replace('\\', '/')
        with open(input_file, 'r') as input_fp:
            soup = BeautifulSoup(input_fp.read(), 'html.parser')
            h1 = soup.find_all('h1')
            h3 = soup.find_all('h3')
            p = soup.find_all('p')
            logger.info('file: {} size: {} h1: {} h3: {} p: {}'.format(input_file, len(soup.text), len(h1), len(h3),
                                                                       len(p)))
            name = h1[1].contents[0].strip()
            missing = 0
            found = 0
            t = p[1].find('h3')
            if hasattr(t, 'contents') and len(t.contents):
                logger.info('{} : {}'.format(name, t.contents[0]))
                found += 1
            else:
                missing += 1

    logger.info('found/missing: {}/{}'.format(found, missing))
    logger.info('input file count: {}'.format(len(input_files)))
