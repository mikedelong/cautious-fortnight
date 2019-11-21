import logging
from time import time

import matplotlib.pyplot as plt
import torch
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
    time_start = time()
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    logger.info(tokenizer)

    text = 'After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank.'
    marked_text = '[CLS] ' + text + ' [SEP]'

    logger.info(marked_text)

    tokenized_text = tokenizer.tokenize(marked_text)
    logger.info(tokenized_text)

    logger.info(list(tokenizer.vocab.keys())[5000:5020])

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    for tup in zip(tokenized_text, indexed_tokens):
        logger.info(tup)

    segments_ids = [1] * len(tokenized_text)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')

    # Put the model in 'evaluation' mode, meaning feed-forward operation.
    logger.info(model.eval())

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    layer_i = 0
    batch_i = 0
    token_i = 0
    format_string = 'layers: {} batches: {} tokens: {} hidden units: {}'
    logger.info(format_string.format(len(encoded_layers), len(encoded_layers[layer_i]),
                                     len(encoded_layers[layer_i][batch_i]),
                                     len(encoded_layers[layer_i][batch_i][token_i])))

    # For the 5th token in our sentence, select its feature values from layer 5.
    token_i = 5
    layer_i = 5
    vec = encoded_layers[layer_i][batch_i][token_i]

    # Plot the values as a histogram to show their distribution.
    plt.figure(figsize=(10, 10))
    plt.hist(vec, bins=200)
    bins_output_file = './output/embeddings_bins.png'
    logger.info('graphing bins to {}'.format(bins_output_file))
    plt.savefig(bins_output_file)

    # Convert the hidden state embeddings into single token vectors

    batch_0 = 0
    # for batch 0, for each layer look up the vector for each token in the tokenized text
    token_embeddings = [[encoded_layers[layer][batch_0][token] for layer in range(len(encoded_layers))] for token in
                        range(len(tokenized_text))]

    # Sanity check the dimensions:
    logger.info('tokens in sequence: {}'.format(len(token_embeddings)))
    logger.info('layers/token: {}'.format(len(token_embeddings[0])))

    # For each token in the sentence concatenate the vectors (that is, append them together) from the last four layers.
    # Each layer vector is 768 values, so `cat_vec` is length 3,072.

    token_vecs_cat = [torch.cat((token[-1], token[-2], token[-3], token[-4]), 0) for token in token_embeddings]
    logger.info('Shape is: {} x {}'.format(len(token_vecs_cat), token_vecs_cat[0].shape[0]))

    # for each token in the sentence, sum the vectors from the last four layers
    token_vecs_sum = [torch.sum(torch.stack(token)[-4:], 0) for token in token_embeddings]

    logger.info('Shape is: {} x {}'.format(len(token_vecs_sum), token_vecs_sum[0].shape[0]))

    sentence_embedding = torch.mean(encoded_layers[11], 1)
    logger.debug(sentence_embedding)

    logger.info('First few values of \'bank\' as in \'bank robber\': {}'.format(token_vecs_sum[10][:8]))

    for index, word in enumerate(tokenized_text):
        if word == 'bank':
            logger.info(token_vecs_sum[index][:8])

    # Compare 'bank' as in 'bank robber' to 'bank' as in 'river bank'
    logger.info('bank robber vs. river bank: {:5.3f}'.format(
        cosine_similarity(token_vecs_sum[10].reshape(1, -1), token_vecs_sum[19].reshape(1, -1))[0][0]))

    # Compare 'bank' as in 'bank robber' to 'bank' as in 'bank vault'
    logger.info('bank robber vs. bank vault: {:5.3f}'.format(cosine_similarity(token_vecs_sum[10].reshape(1, -1),
                                                                               token_vecs_sum[6].reshape(1, -1))[0][0]))
