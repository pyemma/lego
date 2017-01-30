from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np


def get_banknote_data():
    dataset = np.loadtxt('dataset/data_banknote_authentication.csv', delimiter=',')
    shuffled_index = np.arange(dataset.shape[0])
    np.random.shuffle(shuffled_index)
    training_data = dataset[shuffled_index[:1000], 0 : dataset.shape[1] - 1]
    training_labels = dataset[shuffled_index[:1000], dataset.shape[1] - 1]
    testing_data = dataset[shuffled_index[1000:], 0 : dataset.shape[1] - 1]
    testing_label = dataset[shuffled_index[1000:], dataset.shape[1] - 1]
    return training_data, training_labels, testing_data, testing_label
