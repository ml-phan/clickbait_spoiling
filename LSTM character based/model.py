""" This is a character based LSTM model for the clickbait challenge."""

import torch
import torch.nn as nn

from char_based_lstm.utils import STRINGS, NUM_SYMBOLS
from char_based_lstm.utils import load_data, get_symbol_idx, one_hot_vector, create_input_matrix, random_clickbait


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


clickbait_data, clickbait_type = load_data()
class_list = list(set(clickbait_type.values()))
num_classes = len(class_list)

print(num_classes)
n_hidden = 128
rnn = RNN(NUM_SYMBOLS, n_hidden, num_classes)

# one step
input_tensor = one_hot_vector('A')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor, hidden_tensor)
print(output.size())
print(next_hidden.size())

# whole sequence
input_tensor = create_input_matrix('Haus')
hidden_tensor = rnn.init_hidden()

output, next_hidden = rnn(input_tensor[0], hidden_tensor)
print(output.size())
print(next_hidden.size())


def class_from_output(output):
    class_idx = torch.argmax(output).item()
    return class_list[class_idx]


print(class_from_output(output))


criterion = nn.NLLLoss()
learning_rate = 0.005
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)


def train(input_matrix, category_tensor):
    hidden = rnn.init_hidden()

    for i in range(input_matrix.size()[0]):
        output, hidden = rnn(input_matrix[i], hidden)

    loss = criterion(output, category_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()


current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iter = 100
# n_iter = 100000

for i in range(n_iter):
    input_matrix, class_matrix, target_class, target_text = random_clickbait(clickbait_data, clickbait_type)

    output, loss = train(input_matrix, class_matrix)
    current_loss += loss

    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0

    if (i+1) % print_steps == 0:
        guess = class_from_output(output)
        correct = "CORRECT" if guess == target_class else f"WRONG ({target_class})"
        print(f"{i+1} {(i+1)/n_iter*100} {loss:.4f} {target_text} / {guess} {correct}")

def predict(text):

    print(f"\n> {text}")

    with torch.no_grad():
        input_matrix = create_input_matrix(text)
        hidden = rnn.init_hidden()

        for i in range(input_matrix.size()[0]):
            output, hidden = rnn(input_matrix[i], hidden)

        guess = class_from_output(output)
        print(guess)


while True:
    sentence = input("Input:")
    if sentence == "quit":
        break

    predict(sentence)
