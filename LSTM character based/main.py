""" This main file sets the hyperparameters and calls different modules to
ceate the pipeline. """

import torch

from utils import NUM_SYMBOLS, load_data, create_input_matrix, data_handler
from model import LSTM
from train import training

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# hyperparameters
data, types = load_data(True)
test_data, test_types = load_data(False)
class_list = list(set(types.values()))

# parameter data
INPUT_SIZE = NUM_SYMBOLS
NUM_CLASSES = len(class_list)
TRAIN_SIZE = len(data)

# parameter model
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 1

# parameter training
NUM_EPOCHS = 1
# LEARNING_RATE = 0.005
LEARNING_RATE = 0.01


def magic():
    """ Creates and trains the model. """

    model = LSTM(NUM_SYMBOLS, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, device).to(device)
    data_generator = data_handler(data, types)

    training(model, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, device, TRAIN_SIZE, data_generator)
    return model


def predict(model, text):
    """ Lets the model make a prediction on a string. """

    model.eval()
    with torch.no_grad():
        input_matrix = create_input_matrix(text)
        output = model(input_matrix)
        _, predicted = torch.max(output.data, 1)
        return (class_list[predicted])


def evaluation(model):
    """ Calls predict for all data samples and calculates accuracy. """

    total = 0
    correct = 0

    for key, text in test_data.items():
        prediction = predict(model, text)
        total += 1
        correct += 1 if prediction == test_types[key] else 0
        accuracy = 100 * correct/total

        if total % 25 == 0:
            print(f'Total Samples: {total}; Accuracy: { "{:.2f}".format(accuracy)} %')


def train_evaluation(model):
    """ Same as evaluation but uses the training set. """

    total = 0
    correct = 0

    for key, text in data.items():
        input_matrix = create_input_matrix(text)
        prediction = model.predict(input_matrix)
        total += 1
        correct += 1 if prediction == types[key] else 0
        accuracy = 100 * correct/total

        if total % 100 == 0:
            print(f'Total Samples: {total}; Accuracy: { "{:.2f}".format(accuracy)} %')
