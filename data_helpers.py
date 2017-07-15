__author__ = 'PC-LiNing'

import numpy as np


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    len(data) is not divisible by batch_size .
    """
    data = np.array(data)
    data_size = len(data)
    if len(data) % batch_size == 0:
        num_batches_per_epoch = int(len(data)/batch_size)
    else:
        num_batches_per_epoch = int(len(data)/batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            # shuffle_indices = train_shuffle[epoch]
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]

        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]