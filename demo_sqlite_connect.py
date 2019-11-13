import logging
import sqlite3
from time import time

if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info('started')

    database_name = 'enwiki.db'
    database_name = 'db.sqlite3'
    connection = sqlite3.connect(database_name)
    cursor = connection.cursor()
    tables = list()
    for row in cursor.execute('''select name from sqlite_master where type = 'table';'''):
        logger.info('our database has table {}'.format(row[0]))
        tables.append(row[0])

    for table in tables:
        cursor.execute('''select * from {}; '''.format(table))
        names = [description[0] for description in cursor.description]
        logger.info('table {} has columns {}'.format(table, names))

        for row in cursor.execute('''select count(*) from {};'''.format(table)):
            logger.info('table {} has {} rows'.format(table, row[0]))

        for row in cursor.execute('''select * from {} limit 10;'''.format(table)):
            logger.info('{}'.format(row))

    # for index, row in enumerate(cursor.execute('''select * from documents;''')):
    #     logger.info('{} {} {}'.format(index, row[0], row[1][:80].replace('\n', ' ')))

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
