""" This is a character based LSTM model for the clickbait challenge."""

import torch
import torch.nn as nn

from char_based_lstm.utils import STRINGS, NUM_SYMBOLS
from char_based_lstm.utils import load_data, get_symbol_idx, one_hot_vector, create_input_matrix, random_clickbait

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters

clickbait_data, clickbait_type = load_data()
class_list = list(set(clickbait_type.values()))
num_classes = len(class_list)

learning_rate = 0.005
#input_size = NUM_SYMBOLS
hidden_size = 128

batch_size = 100
num_epochs = 2
num_layers = 2


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # maybe add: batch_frist=True

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, h0, c0)
        # out: batch_size, seq_length, hidden_size

        out = out[:, -1, :]

        out = self.fc(out)

        return out


# Define Model
model = LSTM(NUM_SYMBOLS, hidden_size, num_layers, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training
def train(input_matrix, class_matrix):

    # Forward pass
    output = model(input_matrix)
    loss = criterion(output, class_matrix)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss


n_total_steps = 100
for epoch in range(num_epochs):
    for i in range(100):
        input_matrix, class_matrix, target_class, target_text = random_clickbait(clickbait_data, clickbait_type)

        input_matrix = input_matrix.to(device)
        class_matrix = class_matrix.to(device)
        output, loss = train(input_matrix, class_matrix)

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
