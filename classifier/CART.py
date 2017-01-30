from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class InternalNode(object):

    def __init__(self, feature, value, left_child, right_child):
        self._feature = feature
        self._value = value
        self._left_child = left_child
        self._right_child = right_child

    def next(self, data):
        """Return either left child or right child given an input data.

        Args:
            data: A Dx1 vector.
        Returns:
            Either a InternalNode or a LeafNode
        """
        data_value = data[self._feature]
        if (data_value <= self._value):
            return self._left_child
        else:
            return self._right_child

    def as_dict(self):
        """Dump the InternalNode into a dictionary version for pretty print.
        """
        return {
            "feature": self._feature,
            "value": self._value,
            "left_child": self._left_child.as_dict(),
            "right_child": self._right_child.as_dict(),
        }

class LeafNode(object):

    def __init__(self, predict):
        self._predict = predict

    def next(self, data):
        """Return the predict.

        Args:
            data: A Dx1 vector.
        Returns:
            A double predict value.
        """
        return self._predict

    def as_dict(self):
        """Dump the LeafNode into a dictionary version for pretty print.
        """
        return {
            "predict": self._predict,
        }

def best_split(data, labels, index):
    """Find the best split for feature data[index]

    This function will use sum squared loss to find the best point to split the data

    Args:
        data: A NxD matrix
        labels: A Nx1 vector
        index: The index of the feature
    Returns:
        A tuple with value, loss, left_split and right_split.
        value: The threshold to split the feature.
        loss: The sum squared loss of best value of the feature.
        left_split: An array of the index of data less than or equal to value.
        right_split: An array of the index of data greater than.
    """
    values = data[:, index]
    sorted_index = np.argsort(values).astype(np.int)
    left_split, best_left_split, right_split, best_right_split = [], [], list(sorted_index), list(sorted_index)
    best_value, best_loss = 0, np.sum((labels - np.mean(labels)) ** 2)

    for idx, ind in enumerate(sorted_index):
        left_split = sorted_index[:idx+1]
        right_split = sorted_index[idx+1:]
        loss = np.sum((labels[left_split] - np.mean(labels[left_split])) ** 2) + np.sum((labels[right_split] - np.mean(labels[right_split])) ** 2)
        if loss < best_loss:
            best_value = values[ind]
            best_loss = loss
            best_left_split, best_right_split = list(left_split), list(right_split)

    return best_value, best_loss, best_left_split, best_right_split

def best_feature(data, labels):
    """Find the feature to split

    Args:
        data: A NxD matrix
        labels: A Nx1 vector
    Returns:
        A tuple with the index of feature, value, left split and right split
    """
    best_feature, best_feature_value, best_left_split, best_right_split = 0, 0, None, None
    best_loss = np.sum((labels - np.mean(labels)) ** 2)
    for i in range(0, data.shape[1]):
        value, loss, left_split, right_split = best_split(data, labels, i)
        if loss < best_loss:
            best_feature = i
            best_feature_value = value
            best_loss = loss
            best_left_split = left_split
            best_right_split = right_split
    return best_feature, best_feature_value, best_left_split, best_right_split

class CART(object):
    """Classification and Regression Tree

    Attributes:
        _root: The root node of the tree
        _max_depth: The maximum depth of the tree
        _min_node: The minimum number of data to split a node
    """

    def __init__(self, max_depth, min_node):
        self._root = None
        self._max_depth = max_depth
        self._min_node = min_node

    def train(self, data, labels):
        self._root = self._split(data, labels, 1)

    def predict(self, data):
        return np.array([self._iter(self._root, row) for row in data])

    def _split(self, data, labels, depth):
        """Recursively split a InternalNode and build the tree.

            The function will check to see if it has already hit the max depth,
            if so, it will directly return a LeafNode. It also check to see if
            we have enought data to build a InternalNode, otherwise it will
            return a LeafNode.
        Args:
            data: A NxD matrix.
            labels: A Nx1 vector.
            depth: Current depth of the node.
        Returns:
            Either a InternalNode or a LeafNode
        """
        if depth >= self._max_depth:
            return self._build_leaf(data, labels)

        if data.shape[0] < self._min_node:
            return self._build_leaf(data, labels)

        feature, value, left_split, right_split = best_feature(data, labels)
        left_child = self._split(data[left_split, :], labels[left_split], depth+1)
        right_child = self._split(data[right_split, :], labels[right_split], depth+1)
        return InternalNode(feature, value, left_child, right_child)

    def _build_leaf(self, data, labels):
        return LeafNode(np.mean(labels))

    def _iter(self, root, row):
        if isinstance(root, LeafNode):
            return root.next(row)

        return self._iter(root.next(row), row)

    def as_dict(self):
        return self._root.as_dict()
