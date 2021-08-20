from code.utils import load_data, trainIters
from code.model import model, model_optimizer, criterion


trainingData = load_data()

MAX_LENGTH = 0
for i in range(len(trainingData)):
    MAX_LENGTH = trainingData[i].get('maxLength')

model = trainIters(
    model,
    model_optimizer,
    criterion,
    trainingData,
    50000,
    int(MAX_LENGTH)
    )
