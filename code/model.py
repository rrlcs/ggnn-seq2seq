import torch
import torch.nn as nn
import torch.optim as optim
from code.seq2seq import Seq2Seq
from code.ggnn_encoder import GGNN_Encoder
from code.rnn_encoder import EncoderRNN
from code.attention_network import Attention
from code.rnn_decoder_with_attention import Decoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


constraint_encoding = dict({
    'SOS': 0, 'EOS': 1, 'f': 2, 'var1': 3, 'var2': 4,
    '=': 5, '<': 6, '>': 7, '<=': 8, '>=': 9, 'and': 10,
    'or': 11, 'not': 12, '=>': 13, '+': 14, '-': 15,
    '*': 16, 'div': 17, '(': 18, ')': 19, 'num': 20})


def def_model(ggnn_encoder, encoder, decoder, learning_rate):
    model = Seq2Seq(ggnn_encoder, encoder, decoder)
    model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    return model, model_optimizer, criterion


hidden_size = 128
learning_rate = 1e-5
MAX_LENGTH = 127

ggnn_encoder = GGNN_Encoder(
    hidden_size=128,
    num_edge_types=1,
    layer_timesteps=[5]
    ).to(device)

encoder = EncoderRNN(21, hidden_size).to(device)
attention = Attention(hidden_size, hidden_size, MAX_LENGTH).to(device)
decoder = Decoder(hidden_size, 22, attention, MAX_LENGTH).to(device)

model, model_optimizer, criterion = def_model(
    ggnn_encoder,
    encoder,
    decoder,
    learning_rate
    )
