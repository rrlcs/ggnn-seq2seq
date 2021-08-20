import torch
import time
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ast_node_encoding = dict(
    {'f': 0, 'var1': 1, 'var2': 2, '=': 3, '<': 4, '>': 5, '<=': 6, '>=': 7,
        'and': 8, 'or': 9, 'not': 10, '=>': 12, '+': 13, '-': 14,
        '*': 15, 'div': 16, 'PAD1': 17, 'PAD2': 18, 'PAD3': 19,
        'PAD4': 20, 'PAD5': 21}
)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("train_lossplot.png")


# Checkpoints
def save_checkpoint(state, filename="training_checkpoint.path.ptor"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)


def load_data():
    with open("data/dataset_multimodal.json", "r") as f:
        data = json.load(f)
    trainingData = data.get('TrainingExamples')

    return trainingData


def train(
    input_tensor,
    target_tensor,
    featureMatrix,
    edgeListofTuples,
    num_of_nodes,
    model,
    model_optimizer,
    criterion,
    max_length
):

    encoder_hidden = model.encoder.initHidden()

    model_optimizer.zero_grad()
    # decoder_optimizer.zero_grad()

    loss, target_length = model(
        input_tensor,
        target_tensor,
        featureMatrix,
        edgeListofTuples,
        num_of_nodes,
        max_length,
        encoder_hidden,
        criterion
        )

    loss.backward()

    model_optimizer.step()
    # decoder_optimizer.step()

    return loss.item() / target_length


# MAX_LENGTH = 111


def trainIters(
    model,
    model_optimizer,
    criterion,
    trainingData,
    n_iters,
    MAX_LENGTH,
    print_every=100,
    plot_every=100
):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    # encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # model = Seq2Seq(ggnn_encoder, encoder, decoder)
    # model, model_optimizer, criterion = def_model(
    #     ggnn_encoder,
    #     encoder,
    #     decoder
    #     )
    # model_optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    # criterion = nn.NLLLoss()
    # print("len: ",len(training_pairs))
    model.train()
    for iter in range(1, n_iters + 1):
        input_tensor = torch.from_numpy(
            np.array(trainingData[iter - 1].get('constraintTokenList'))
            ).to(dtype=torch.long)
        # print(input_tensor.size())
        target_tensor = torch.from_numpy(
            np.array(trainingData[iter - 1].get('programTokenList'))
            ).to(dtype=torch.long)
        # print(target_tensor)

        # Inputs for GGNN
        featureMatrix = trainingData[iter - 1].get('featureMatrix')
        num_of_nodes = trainingData[iter - 1].get('num_of_nodes')
        featureMatrix = torch.tensor(
            np.reshape(
                    np.array(featureMatrix),
                    (
                        num_of_nodes,
                        len(ast_node_encoding)+1
                    )
                ), dtype=torch.float).to(device)
        edgeList = trainingData[iter - 1].get('edgeList')
        edgeListofTuples = list(map(tuple, edgeList))

        loss = train(
            input_tensor,
            target_tensor,
            featureMatrix,
            edgeListofTuples,
            num_of_nodes,
            model,
            model_optimizer,
            criterion,
            max_length=MAX_LENGTH
            )

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter,
                                         iter / n_iters * 100,
                                         print_loss_avg))

            # Get model checkpoint
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': model_optimizer.state_dict()
                }

            # Save checkpoint
            save_checkpoint(checkpoint)

            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

    return model
