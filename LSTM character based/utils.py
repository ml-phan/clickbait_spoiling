""" This is a list of helper functions for the character based LSTM model."""

import json
import string
import torch
import unicodedata


# define all avaiable characters
STRINGS = string.printable
NUM_SYMBOLS = len(STRINGS)


def load_data(train):
    """ Loads data from jsonl-file and saves the important entries"""

    # Test oder validation file
    if train is True:
        file = './data/train.jsonl'
    else:
        file = './data/validation.jsonl'

    with open(file) as jsonl_file:
        jsonl_file = list(jsonl_file)

    data = {}
    types = {}
    # insert the data fields used for training later
    used_fields = ['postText', 'targetTitle']
    # used_fields = ['postText', 'targetParagraphs', 'targetTitle']

    # fill two dicts data and types from the jsonl-file
    for entry in jsonl_file:
        jl_entry = json.loads(entry)
        # all fields are concatenated as strings
        cat_fields = ""
        for field in used_fields:
            data_string = unicode_to_ascii(str(jl_entry[field]))
            cat_fields += data_string

        data.update({jl_entry['uuid']: cat_fields})
        types.update({jl_entry['uuid']: jl_entry['tags'][0]})

    return data, types


def unicode_to_ascii(s):
    """ Turns unicode strings into ascii in order to prevent errors. """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in STRINGS)


def create_input_matrix(text):
    """ Turns a string into a matrix of one-hot vectors. """

    matrix = torch.zeros(len(text), 1, NUM_SYMBOLS)
    for idx, symbol in enumerate(text):
        matrix[idx][0][STRINGS.index(symbol)] = 1
    return matrix


def data_handler(data, types):
    """ Yields the input matrix and gold label for all data. """

    for key, text in data.items():
        # creating input matrix
        clickbait_matrix = create_input_matrix(text)

        # creating gold label
        target_class = types[key]
        class_list = list(set(types.values()))
        gold_label = torch.tensor([class_list.index(target_class)], dtype=torch.long)

        yield clickbait_matrix, gold_label
