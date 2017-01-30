from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from classifier.CART import CART
import numpy as np

class GBDT(object):
    """Gradient Boosting Decision Tree.

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
        self._weight = []  # The weight for each individual tree

    def train(self, data, labels):
        self._train(data, labels)

    def _train(self, data, labels):
        weight = 0.3
        prev = np.zeros(labels.shape)
        for i in range(self._num_iters):
            gradient = self._get_gradient(prev, labels)
            tree = CART(max_depth=self._max_depth, min_node=self._min_node)
            tree.train(data, gradient)
            self._trees.append(tree)
            prev = prev - tree.predict(data) * weight
            self._weight.append(weight)

    def _get_gradient(self, prev, labels):
        """Return the gradient of the loss function for GBDT.

            Currently, I only provide a simple squared sum loss L = sum(labels - prev) ** 2
        Args:
            prev: A Nx1 vector, the output of current classifier h(t-1)
            labels: A Nx1 vector
        Return:
            The gradient.
        """
        return 2 * (prev - labels)

    def predict(self, data):
        result = np.zeros(data.shape[0])
        for tree, weight in zip(self._trees, self._weight):
            result -= tree.predict(data) * weight
        return result
