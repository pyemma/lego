from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from classifier.CART import CART
import numpy as np

class RandomForest(object):
    """Random Forest.

    A simple version random forest. Random select a subset of training data and
    then train a CART on it. We use the accu of the tree on validation data to
    compute the weight for it.

    Args:
        _trees: A list of decision trees.
        _num_iters: Number of trees to train.
        _max_depth: Max depth of each tree.
        _min_node: Minimum number of data to split a node.
    """

    def __init__(self, num_iters, max_depth, min_node):
        self._trees = []
        self._num_iters = num_iters
        self._max_depth = max_depth
        self._min_node = min_node
        self._subsample_ratio = 0.5
        self._weight = []  # The weight for each individual tree
        self._accu = []  # The accu of each tree on validation set

    def train(self, data, labels):
        self._train(data, labels)

    def _train(self, data, labels):
        for i in range(self._num_iters):
            sub_training_data, sub_training_labels, val_data, val_labels = self._subsample(data ,labels)
            tree = CART(max_depth=self._max_depth, min_node=self._min_node)
            tree.train(sub_training_data, sub_training_labels)
            self._trees.append(tree)
            predict = tree.predict(val_data)
            accu = np.mean((predict > 0.5).astype(np.float32) == val_labels)
            self._accu.append(accu)
        self._get_weight()
        print(self._weight)

    def _subsample(self, data, labels):
        num_sample = data.shape[0]
        shuffled_index = np.arange(num_sample)
        np.random.shuffle(shuffled_index)
        subsample_index = shuffled_index[:int(num_sample * self._subsample_ratio)]
        validation_index = shuffled_index[int(num_sample * self._subsample_ratio):]
        return data[subsample_index, :], labels[subsample_index], \
            data[validation_index, :], labels[validation_index]

    def _get_weight(self):
        self._weight = [accu / np.sum(self._accu) for accu in self._accu]

    def predict(self, data, mode='vote'):
        result = np.zeros((self._num_iters, data.shape[0]))
        for i in range(self._num_iters):
            result[i, :] = self._trees[i].predict(data)
        if mode == 'vote':
            result = (result > 0.5).astype(np.float32)
            return np.mean(result, axis=0) > 0.5
        return np.sum(result * self._weight)
