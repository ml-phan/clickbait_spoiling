""" This main file sets the hyperparameters and calls different modules to
ceate the pipeline. """
import json
import time

import torch

from utils import NUM_SYMBOLS, load_data, create_input_matrix, data_handler
from model import LSTM
from train import training

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
# load data
data, types = load_data(True)
test_data, test_types = load_data(False)
class_list = list(set(types.values()))

# parameter data
INPUT_SIZE = NUM_SYMBOLS
NUM_CLASSES = len(class_list)
TRAIN_SIZE = len(data)

# parameter model
HIDDEN_SIZE = 32
NUM_LAYERS = 2
BATCH_SIZE = 1

# parameter training
NUM_EPOCHS = 10
LEARNING_RATE = 0.005


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
        accuracy = 100 * correct / total

    print(f'Total Samples: {total}; Accuracy: {"{:.2f}".format(accuracy)} %')
    return accuracy


def make_model(hidden_size, learning_rate):
    """ The main function. Takes a hidden_size and learning_rate and creates a model with these. Trains the model on the training data and evaluates the model on the validation data. """

    # create a model
    model = LSTM(NUM_SYMBOLS, hidden_size, NUM_LAYERS, NUM_CLASSES, device).to(device)

    # train it
    training(model, BATCH_SIZE, NUM_EPOCHS, learning_rate, device, TRAIN_SIZE, data_handler, data, types)

    # evaluate it
    accuracy = evaluation(model)

    return model, accuracy


def main():
    """Returns the model with the best hyperparameters and its accuracy. """
    learning_rates = [64, 96, 128, 256, 512]
    hidden_dims = [0.0005, 0.001, 0.005]
    results = {}
    for learning_rate in learning_rates:
        for hidden_size in hidden_dims:
            start = time.perf_counter()
            results[f"{learning_rate}_{hidden_size}"] = {}
            print(f"Learning rate {learning_rate}")
            print(f"Hidden Size {hidden_size}")
            _, results[f"{learning_rate}_{hidden_size}"]["Accuracy"] = make_model(learning_rate, hidden_size)
            end = time.perf_counter()
            results[f"{learning_rate}_{hidden_size}"]["Training Time"] = start - end

    with open("lstm_results.txt", "w") as file:
        file.write(json.dumps(results, indent=4))


if __name__ == '__main__':
    main()
