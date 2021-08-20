import torch
import numpy as np
import re
import nltk
from torchtext.data.metrics import bleu_score
from code.ggnn_encoder import AdjacencyList
from code.utils import load_data, ast_node_encoding
from code.model import model, model_optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

program_encoding = dict({
    'SOS': 0, 'EOS': 1, 'f': 2, 'var1': 3, 'var2': 4,
    '=': 5, '<': 6, '>': 7, '<=': 8, '>=': 9, 'and': 10,
    'or': 11, 'not': 12, 'ite': 13, '+': 14, '-': 15,
    '*': 16, 'div': 17, '(': 18, ')': 19, 'num': 20, 'mod': 21})

SOS_token = 0
EOS_token = 1

trainingData = load_data()


# Load saved checkpoint
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def evaluate(
    model,
    featureMatrix,
    num_of_nodes,
    edgeListofTuples,
    sentence,
    max_length
):

    model.eval()
    with torch.no_grad():
        input_tensor = sentence
        input_length = input_tensor.size()[0]
        encoder_hidden = model.encoder.initHidden()

        # loss, target_length = model(input_tensor, target_tensor
        # , featureMatrix, edgeListofTuples, num_of_nodes, max_length,
        #  encoder_hidden, criterion)
        encoder_outputs = torch.zeros(
            max_length,
            model.encoder.hidden_size,
            device=device
            )

        adj_list_type1 = AdjacencyList(
            node_num=num_of_nodes,
            adj_list=edgeListofTuples,
            device=device
            )

        graph_repr = model.ggnn_encoder(
            featureMatrix,
            adjacency_lists=[adj_list_type1]
            )
        graph_repr = graph_repr.view(1, 1, -1)

        for ei in range(input_length):
            encoder_output, encoder_hidden = model.encoder(
                graph_repr,
                input_tensor[ei],
                encoder_hidden
                )

            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        mask = model.create_mask(encoder_outputs)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = model.decoder(
                decoder_input, decoder_hidden, encoder_outputs, mask)
            # print(decoder_attention.size())
            decoder_attentions[di] = decoder_attention.permute(1, 0)[0].data
            topv, topi = decoder_output.data.topk(1)
            # print(topv, topi)
            if topi.item() == EOS_token:
                decoded_words.append('EOS')
                break
            else:
                decoded_words.append(
                    list(program_encoding.keys())[list(
                        program_encoding.values()
                        ).index(topi.item())]
                    )

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


test_data_size = 30000
start_test_idx = 50001
end_test_idx = start_test_idx + test_data_size


def evaluateRandomly(model, n=test_data_size):
    candidate_sequences = []
    reference_corpus = []
    bleu_scores = 0
    for i in range(start_test_idx, end_test_idx):
        reference_sequences = []
        print('>', trainingData[i].get('constraint'))
        print('=', trainingData[i].get('program'))

        # Inputs for GGNN
        featureMatrix = trainingData[i].get('featureMatrix')
        num_of_nodes = trainingData[i].get('num_of_nodes')
        featureMatrix = torch.tensor(
            np.reshape(
                    np.array(featureMatrix),
                    (
                        num_of_nodes,
                        len(ast_node_encoding)+1
                    )
                ), dtype=torch.float).to(device)
        edgeList = trainingData[i].get('edgeList')
        edgeListofTuples = list(map(tuple, edgeList))

        output_words, attentions = evaluate(
            model,
            featureMatrix,
            num_of_nodes,
            edgeListofTuples,
            torch.from_numpy(
                np.array(
                    trainingData[i].get(
                        'constraintTokenList'
                        )
                    )
                ).to(dtype=torch.long),
            max_length=111
            )

        output_sentence = ' '.join(output_words)
        print('<', output_sentence)

        candidate_sequences.append(output_words)
        target = trainingData[i].get('program')
        target = target.replace('(', '( ').replace(')', ' )')
        target = re.sub('\d', 'num', target)
        if target.count('x') > 0:
            target = target.replace('x', 'var1')
        if target.count('y') > 0:
            target = target.replace('y', 'var2')
        target = 'SOS '+target+' EOS'
        target_tokens = nltk.tokenize.wordpunct_tokenize(target)
        reference_sequences.append(target_tokens)
        reference_corpus.append(reference_sequences)

        bleu_scores += round(
            bleu_score(candidate_sequences, reference_corpus),
            2
            )
        print(f'BLEU Score: {round(bleu_score(candidate_sequences, reference_corpus), 2)}')
        print('')

    # print(candidate_sequences)
    # print(reference_sequences)
    # reference_corpus = [reference_sequences, [['Another', 'Sentence']]]
    # print(references_corpus)
    print(f'Average BLEU Score: {bleu_scores/n}')


# Load saved checkpoint
load_checkpoint(
    torch.load('training_checkpoint.path.ptor'),
    model,
    model_optimizer
    )

evaluateRandomly(model)
