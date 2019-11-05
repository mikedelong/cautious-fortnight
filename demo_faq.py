from deeppavlov import configs
from deeppavlov.core.commands.infer import build_model

if __name__ == '__main__':
    model = build_model(configs.faq.tfidf_logreg_en_faq, load_trained=True)
    print(model)
