from deeppavlov import configs
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.common.file import read_json

if __name__ == '__main__':
    configuration = read_json(configs.faq.tfidf_logreg_en_faq)
    faq = build_model(config=configuration, download=True, )
    question = 'I need help'
    answer = faq([question])[0][0]
    print('Q: {} A: {}'.format(question, answer))
