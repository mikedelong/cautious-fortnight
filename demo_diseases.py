import logging
from time import time

from deeppavlov import configs
from deeppavlov import train_model
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.file import read_json

if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info('started')

    case = 'arxiv'
    cases = {
        'arxiv': './SentenceCorpus/unlabeled_articles/arxiv_unlabeled',
        'plos': './SentenceCorpus/unlabeled_articles/plos_unlabeled',
    }
    model_config = read_json(configs.doc_retrieval.en_ranker_tfidf_wiki)
    model_config['dataset_reader']['data_path'] = cases[case]
    model_config['dataset_reader']['dataset_format'] = 'txt'
    doc_retrieval = train_model(model_config)

    logger.info(doc_retrieval(['cerebellum']))

    # Download all the SQuAD models
    squad = build_model(configs.squad.multi_squad_noans_infer, download=True)
    # Do not download the ODQA models, we've just trained it
    model = build_model(configs.odqa.en_odqa_infer_wiki, download=False)
    plos_questions = sorted([
        'What is rubella?',
        'What is whooping cough?',
        # 'What are yams?',
        'What are yaws?',
        # 'What are cars?',
        # 'Where does leather come from?',
        # 'How many eggs are in a dozen?',
        'What is influenza?',
        'What is measles?',
        # 'What is arthritis?'
        'What is marginalization?',
        'Who was Bernoulli?',
        'Who is or was Bayes?',
        'What is phylogeny?',
        'What is phylogenetic?',
        'What is evolution?',
        'What is protein?',
    ])

    arxiv_questions = sorted([
        'Who was Bernoulli?',
        'Who is or was Bayes?',
        # 'What is a machine?',
        'What is machine learning?',
        # 'What is WAIC?',
        # 'When did it start?',
        # 'What is a forecast?',
        # 'What is a strategy?',
        # 'What is a forecasting strategy?',
        # 'When was Levin?',
        # 'When did Levin?',
        # 'What is symbol?',
        # 'What is SYMBOL?',
        'What is a realization?',
        'What is a robot?',
        'How soon is now?',
        'What is the inverse problem?',
        'What is a problem?',
        'What is the inverse?',
    ])
    questions = plos_questions if case == 'plos' else arxiv_questions
    for question in questions:
        answer = model([question])
        logger.info('Q: {} A: {}'.format(question, answer[0]))

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
