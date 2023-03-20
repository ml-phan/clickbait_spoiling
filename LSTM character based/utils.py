""" This is a list of helper functions for the character based LSTM model."""

import json
import string
import torch
import random
import unicodedata


STRINGS = string.printable
NUM_SYMBOLS = len(STRINGS)


def load_data():
    """ Loads data from jsonl-file and saves the important entries"""

    with open('./data/train.jsonl') as jsonl_file:
        jsonl_file = list(jsonl_file)

    clickbait_data = {}
    clickbait_type = {}
    # insert the data fields used for training later
    used_fields = ['targetTitle']
    # used_fields = ['postText', 'targetParagraphs', 'targetTitle']

    for entry in jsonl_file:
        jl_entry = json.loads(entry)
        # all fields are concatenated as strings
        cat_fields = ""
        for field in used_fields:
            data_string = unicode_to_ascii(str(jl_entry[field]))
            cat_fields += data_string
        clickbait_data.update({jl_entry['uuid']: cat_fields})
        clickbait_type.update({jl_entry['uuid']: jl_entry['tags'][0]})

    return clickbait_data, clickbait_type

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in STRINGS
    )

def get_symbol_idx(symbol):
    return STRINGS.index(symbol)


def one_hot_vector(letter):
    vector = torch.zeros(1, NUM_SYMBOLS)
    vector[0][get_symbol_idx(letter)] = 1
    return vector


def create_input_matrix(text):
    matrix = torch.zeros(len(text), 1, NUM_SYMBOLS)
    for idx, symbol in enumerate(text):
        matrix[idx][0][get_symbol_idx(symbol)] = 1
    return matrix


def random_clickbait(clickbait_data, clickbait_type):

    random_uuid = random.choice(list(clickbait_data.keys()))
    target_class = clickbait_type[random_uuid]
    target_text = clickbait_data[random_uuid]
    class_list = list(set(clickbait_type.values()))

    clickbait_matrix = create_input_matrix(clickbait_data[random_uuid])
    class_matrix = torch.tensor([class_list.index(target_class)], dtype=torch.long)

    return clickbait_matrix, class_matrix, target_class, target_text
