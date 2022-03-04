import numpy as np
import pandas as pd


class BaseTree:
    def __init__(self, criterion, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self._root = None
        self._X = None
        self._y = None
        self._criterion = criterion
        pass

    def fit(self, X, y):
        self._root = Node(X, y)
        self._X = X
        self._y = y
        pass

    def predict(self, X):
        pass

    def should_stop(self, node):
        # checking for max_depth, min_samples_split, min_samples_leaf
        return False

    def split_node(self, node):
        column, threshold = node.find_best_split(self._criterion)
        if self.should_stop(node):
            return

        data = node.get_data()
        labels = node.get_labels()
        is_smaller = data[:, column] < threshold
        left_data = data[is_smaller, :]
        right_data = data[~is_smaller, :]
        left_labels = labels[is_smaller, :]
        right_labels = labels[~is_smaller, :]
        child_depth = node.get_depth() + 1
        left_node = Node(left_data, left_labels, parent=node, depth=child_depth)
        right_node = Node(right_data, right_labels, parent=node, depth=child_depth)
        node.add_rule(column, threshold)
        node.add_children(left_node=left_node, right_node=right_node)
        self.split_node(left_node)
        self.split_node(right_node)


class Node:
    def __init__(self, X, y, parent=None, depth=0):
        self._data = X
        self._labels = y
        self._num_samples = len(self._labels)
        self._parent = parent
        self._depth = depth
        self._column = None
        self._threshold = None
        self._left_child = None
        self._right_child = None

    def get_depth(self):
        return self._depth

    def get_num_samples(self):
        return self._num_samples

    def get_data(self):
        return self._data

    def get_labels(self):
        return self._labels

    def add_rule(self, column, threshold):
        self._column = column
        self._threshold = threshold

    def add_children(self, left_node, right_node):
        self._left_child = left_node
        self._right_child = right_node

    def compute_gini(self):
        pass

    def compute_cross_entropy(self):
        pass

    def compute_mse(self):
        pass

    def compute_mae(self):
        pass

    def split_by_column(self, column, criterion):  # move to Node
        # will use node.compute_gini() or node.compute_mse()
        column_values = self._data[:, column]
        score = 0
        threshold = 0
        return score, threshold

    def find_best_split(self, criterion):   # move to Node
        best_score = np.inf
        best_threshold = 0
        best_column = None
        for column in self._data.columns:  # iterating over columns
            score, threshold = self.split_by_column(column, criterion)
            if score < best_score:
                best_score = score
                best_threshold = threshold
                best_column = column

        return best_column, best_threshold
