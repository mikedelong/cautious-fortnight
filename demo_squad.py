import logging
from time import time

from deeppavlov import build_model

if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info('started')

    configuration = './squad.json'
    model = build_model(configuration, download=True)
    data = {
        'Apollo 11 was the spaceflight that first landed humans on the Moon. Commander Neil Armstrong and '
        'lunar module pilot Buzz Aldrin formed the American crew that landed the Apollo Lunar Module Eagle on '
        'July 20, 1969, at 20:17 UTC. Armstrong became the first person to step onto the lunar surface six hours '
        'and 39 minutes later on July 21 at 02:56 UTC; Aldrin joined him 19 minutes later. They spent about '
        'two and a quarter hours together outside the spacecraft, and they collected 47.5 pounds (21.5 kg) of '
        'lunar material to bring back to Earth. Command module pilot Michael Collins flew the command module '
        'Columbia alone in lunar orbit while they were on the Moon\'s surface. Armstrong and Aldrin spent 21 hours, '
        '36 minutes on the lunar surface at a site they named Tranquility Base before lifting off to rejoin '
        'Columbia in lunar orbit.': [
            'When was the first moon landing?',
            'When did Eagle land?',
            'What was the Apollo 11 landing site named?',
            'What was the landing site named?',
            'What was the site named?',
            'Who was the Eagle pilot?',
            'Who was the lunar module pilot?',
            'Who flew the command module?',
            'What was the name of the command module?',
            'How much lunar material did Apollo 11 bring back?',
            'How long were Armstrong and Aldrin on the moon?',
            'How long were Armstrong and Aldrin on the lunar surface?',
        ],
        'No. 33 Squadron is a Royal Australian Air Force (RAAF) strategic transport and air-to-air refuelling '
        'squadron. It operates Airbus KC-30A Multi Role Tanker Transports from RAAF Base Amberley, Queensland. '
        'The squadron was formed in February 1942 for service during World War II, operating Short Empire flying '
        'boats and a variety of smaller aircraft. By 1944 it had completely re-equipped with Douglas C-47 Dakota '
        'transports, which it flew in New Guinea prior to disbanding in May 1946. The unit was re-established in '
        'February 1981 as a flight, equipped with two Boeing 707s for VIP and other long-range transport duties out '
        'of RAAF Base Richmond, New South Wales. No. 33 Flight was re-formed as a full squadron in July 1983. By 1988 '
        'it was operating six 707s, four of which were subsequently converted for aerial refuelling. The 707s saw '
        'active service during operations in Namibia, Somalia, the Persian Gulf, and Afghanistan. One of the '
        'transport jets was lost in a crash in October 1991. No. 33 Squadron relocated to Amberley and was '
        'temporarily without aircraft following the retirement of the 707s in June 2008. It began re-equipping '
        'with KC-30As in June 2011, and achieved initial operating capability with the type in February 2013. One of '
        'its aircraft was deployed to the Middle East in September 2014, as part of Australia\'s contribution to the '
        'military coalition against ISIS.': [
            'When was Number 33 Squadron re-established?',
            'How many planes were in Number 33 Squadron?',
            'What kind of planes were in Number 33 Squadron?',
            'Number 33 Squadron was equipped with what kind of planes?',
            'Number 33 Squadron operated what kind of planes?',
            'Number 33 Squadron operates what kind of planes?',
            'When did the squadron lose a plane in a crash?',
            'When did the squadron retire its 707s?',
        ],
    }
    for text, questions in data.items():
        for question in questions:
            logger.info('Q: {} A: {}'.format(question, model([text], [question])))

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
