import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        output_size,
        attention,
        max_length,
        dropout_p=0.1
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attention = attention
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, mask):
        embedded = self.embedding(input).view(1, 1, -1)
        # print(embedded.size())
        embedded = self.dropout(embedded)

        attn_weights = self.attention(hidden[0], encoder_outputs, mask)
        # print("attn_weights: ", attn_weights.unsqueeze(0).size())
        # print("enc_out: ", encoder_outputs.unsqueeze(0).size())
        attn_applied = torch.bmm(attn_weights.unsqueeze(0).permute(0, 2, 1),
                                 encoder_outputs.unsqueeze(0))
        # print("attn_applied: ", attn_applied[0].size())
        # print("embedded: ", embedded[0].size())
        attn_applied = attn_applied.permute(1, 0, 2)
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
