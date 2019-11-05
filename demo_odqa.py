from deeppavlov import configs
from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.train import train_evaluate_model_from_config

if __name__ == '__main__':
    train_evaluate_model_from_config(configs.doc_retrieval.en_ranker_tfidf_wiki, download=True)
    train_evaluate_model_from_config(configs.squad.multi_squad_noans, download=True)
    odqa = build_model(configs.odqa.en_odqa_infer_wiki, load_trained=True)

    result = odqa(['What is the name of Darth Vader\'s son?'])
    print(result)
