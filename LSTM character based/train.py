""" This file contains the training function in order to train the LSTM model.
"""

import torch
import torch.nn as nn


# Training Function
def training(model, batch_size, epochs, learning_rate, device, data_length, data_handler):

    # Define learning devices
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        for n in range(data_length):
            optimizer.zero_grad()

            # get data and gold_labels
            input_matrix, gold_label = next(data_handler)
            input_matrix = input_matrix.to(device)
            gold_label = gold_label.to(device)

            # Forward pass
            output = model(input_matrix)
            loss = criterion(output, gold_label)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            if n % 50 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], n [{n}], Loss: {loss.item():.4f}')
