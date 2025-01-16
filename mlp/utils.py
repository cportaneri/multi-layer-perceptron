import random

def shuffle_data(dataset_inputs, dataset_outputs):
    shuffled_data = list(zip(dataset_inputs, dataset_outputs))
    random.shuffle(shuffled_data)        
    return list(zip(*shuffled_data))