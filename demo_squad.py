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
    data = {
        'Apollo 11 was the spaceflight that first landed humans on the Moon. Commander Neil Armstrong and '
        'lunar module pilot Buzz Aldrin formed the American crew that landed the Apollo Lunar Module Eagle on '
        'July 20, 1969, at 20:17 UTC. Armstrong became the first person to step onto the lunar surface six hours '
        'and 39 minutes later on July 21 at 02:56 UTC; Aldrin joined him 19 minutes later. They spent about '
        'two and a quarter hours together outside the spacecraft, and they collected 47.5 pounds (21.5 kg) of '
        'lunar material to bring back to Earth. Command module pilot Michael Collins flew the command module '
        'Columbia alone in lunar orbit while they were on the Moon\'s surface. Armstrong and Aldrin spent 21 hours, '
        '36 minutes on the lunar surface at a site they named Tranquility Base before lifting off to rejoin '
        'Columbia in lunar orbit.': ['When was the first moon landing?',
                                     'When did Eagle land?',
                                     'What was the Apollo 11 landing site named?',
                                     'What was the landing site named?',
                                     'What was the site named?',
                                     'Who was the lunar module pilot?',
                                     'How much lunar material did Apollo 11 bring back?',
                                     'How long were Armstrong and Aldrin on the moon?',
                                     'How long were Armstrong and Aldrin on the lunar surface?',
                                     ]
    }
    for text, questions in data.items():
        for question in questions:
            logger.info('Q: {} A: {}'.format(question, model([text], [question])))

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
