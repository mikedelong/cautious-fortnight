from deeppavlov.core.commands.infer import build_model

if __name__ == '__main__':
    config_ = './tfidf_logreg_en_faq.json'
    model = build_model(config=config_, download=True, load_trained=True, )
    for question in ['What time is it now?', 'What are your open hours?', ]:
        result = model([question])
        print('Q: {} A: {}'.format(question, result[0][0]))
