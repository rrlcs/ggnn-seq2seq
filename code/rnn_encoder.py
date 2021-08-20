import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # print("inp size: ",input_size)
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size)

    def forward(self, graph_repr, input, hidden):
        # print("input: ", input)
        embedded = self.embedding(input).view(1, 1, -1)
        multimodal = torch.cat((graph_repr, embedded), dim=2)
        # print("multimodel (1,1,512): ", multimodal.size())
        # print("embedded: ", embedded.size())
        # output = embedded
        output = multimodal
        # print("output: ", output.size())
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
