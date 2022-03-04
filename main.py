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
        if max_depth == None: return True
        if node.get_depth() >= max_depth: return True
        if node.get_num_samples() < min_samples_split: return True
        if node.get_num_samples() <= min_samples_leaf: return True

        #if not beyond depth or not below min samples
        return False

    def split_node(self, node):
        column, threshold = node.find_best_split(self._criterion)
        if self.should_stop(node): return

        left_data, right_data, left_labels, right_labels = node.split_node(column, threshold)

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

    def compute_gini(self, left_count, right_count):
        # Ensuring the correct types
        if left_count is None:
            left_count = 0

        if right_count is None:
            right_count = 0

        # Getting the total observations
        n = left_count + right_count
        
        # If n is 0 then we return the lowest possible gini impurity
        if n == 0:
            return 0.0

        # Getting the probability to see each of the classes 
        # assumes only two classes, idk how many are in our data @asror
        p1 = left_count / n
        p2 = right_count / n
        
        # Calculating GINI 
        gini = 1 - (p1 ** 2 + p2 ** 2) 
        return gini

    def compute_cross_entropy(self):
        pass

    def compute_mse(self):
        pass

    def compute_mae(self):
        pass 

    def split_node(self, column, threshold): # moved to Node
        # function for spliting X, y to two nodes
        # according to a given column and threshold
        is_smaller = self._data[:, column] < threshold
        left_data = self._data[is_smaller, :]
        right_data = self._data[~is_smaller, :]
        left_labels = self._labels[is_smaller, :]
        right_labels = self._labels[~is_smaller, :]
        return left_data, right_data, left_labels, right_labels

    def split_by_column(self, column, criterion):  # move to Node
        # will use node.compute_gini() or node.compute_mse() ]
        best_score = np.inf
        best_threshold = None

        # get all data for this column
        # then look for unique values to be our thresholds
        column_values = self._data[:, column]
        unique_values = np.unique(column_values) 

        for threshold in unique_values: #iterate through possible thresholds
            # we take the (distribution of) labels 
             _, _, left, right = self.split_node(column, threshold)
             left_counts, right_counts = Counter(left), Counter(right)

            # Classification or Regression task?
            if criterion == "classification": 
                score = node.compute_gini(left_counts, right_counts)
            if criterion == "regression": 
                score = node.compute_mse(left_counts, right_counts) 

            # improve our scores and move to the next possible threshold
            if score < best_score:
                best_score = score
                best_threshold = threshold

        return best_score, best_threshold

    def find_best_split(self, criterion):   # move to Node
        best_score = np.inf
        best_threshold = 0
        best_column = None 

        for column in self._data.columns:  # iterating over columns
            # find the best score/threshold for this column
            score, threshold = self.split_by_column(column, criterion)

            # improve our scores and move to the next possible column
            if score is not None and score < best_score: 
                best_score = score
                best_threshold = threshold
                best_column = column

        # returns the best column after iterating through possible columns
        # returns the best threshold in the best column
        return best_column, best_threshold
