from deeppavlov.core.commands.infer import build_model

if __name__ == '__main__':
    config_ = './tfidf_logreg_en_faq.json'
    model = build_model(config=config_, load_trained=True)
    result = model(['What time is it now?'])
    print(result)