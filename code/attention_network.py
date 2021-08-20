import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, max_length):
        super().__init__()
        self.max_length = max_length
        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        # batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        # print(hidden.size())
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # permute(1, 0, 2)
        hidden = hidden.squeeze(0)
        # print(hidden.size())
        # view(1, src_len, hidden_size).permute(1, 0, 2) -|
        encoder_outputs = encoder_outputs
        # print(encoder_outputs.size())

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=1))
            )
        # print("energy: ", energy.size())
        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy)  # squeeze(2)
        attention = attention.permute(1, 0)
        # print("attention: ", attention.size())

        # attention = [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)
        attention = attention.permute(1, 0)
        # print("attention: ", attention.size())

        return F.softmax(attention, dim=1)
