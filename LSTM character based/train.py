"""
This file contains the training function in order to train the LSTM model.
This model structure is partly based on the version by Patrick Loeber:
https://github.com/patrickloeber/pytorch-examples/blob/master/rnn-lstm-gru/main.py
"""

import torch
import torch.nn as nn


# Training Function
def training(model, batch_size, epochs, learning_rate, device, data_length, data_generator_func, data, types):

    # Define learning devices
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        data_generator = data_generator_func(data, types)
        losses = 0.0
        for n in range(data_length):
            optimizer.zero_grad()

            # get data and gold_labels
            input_matrix, gold_label = next(data_generator)
            input_matrix = input_matrix.to(device)
            gold_label = gold_label.to(device)

            # Forward pass
            output = model(input_matrix)
            loss = criterion(output, gold_label)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            losses += loss.item()
            if n % 400 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Samples [{n}/{data_length}], Loss: {loss.item():.4f}, Loss_avg = {(losses/(n+1)):.4f}')
