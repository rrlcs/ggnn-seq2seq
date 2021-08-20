import torch
import torch.nn as nn
import random
from code.ggnn_encoder import AdjacencyList
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1


class Seq2Seq(nn.Module):
    def __init__(self, ggnn_encoder, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.ggnn_encoder = ggnn_encoder
        self.encoder = encoder
        self.decoder = decoder

    def create_mask(self, src):
        mask = (src != 0).permute(1, 0)
        # print("mask: ", mask.size())
        return mask

    def forward(
        self,
        input_tensor,
        target_tensor,
        featureMatrix,
        edgeListofTuples,
        num_of_nodes,
        max_length,
        encoder_hidden,
        criterion,
        teacher_forcing_ratio=0
    ):

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(
            max_length,
            self.encoder.hidden_size,
            device=device
            )

        adj_list_type1 = AdjacencyList(
            node_num=num_of_nodes,
            adj_list=edgeListofTuples,
            device=device
            )

        graph_repr = self.ggnn_encoder(
            featureMatrix,
            adjacency_lists=[adj_list_type1]
            )

        graph_repr = graph_repr.view(1, 1, -1)
        # print("graph repr: ", graph_repr.size())
        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                graph_repr,
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if (
            random.random() < teacher_forcing_ratio
            ) else False

        # print("input tensor: ", input_tensor)
        mask = self.create_mask(encoder_outputs)
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention =\
                     self.decoder(
                         decoder_input, decoder_hidden, encoder_outputs, mask)
                # print("decoder_output: ", decoder_output)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing:
            # use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention =\
                     self.decoder(
                         decoder_input, decoder_hidden, encoder_outputs, mask)
                topv, topi = decoder_output.topk(1)

                # detach from history as input
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        return loss, target_length
