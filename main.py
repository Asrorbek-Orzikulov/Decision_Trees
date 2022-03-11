import numpy as np
import pandas as pd


class BaseTree:
    def __init__(self, criterion, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self._root = None
        self._X = None
        self._y = None
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        pass

    def fit(self, X, y):
        self._root = Node(X, y)
        self._X = X
        self._y = y
        pass

    def predict(self, X):
        pass

    def should_stop(self, node):
        # checking for max_depth and min_samples_split
        if self._max_depth is None:
            return False  # don't agree
        elif node.get_depth() >= self._max_depth:
            return True
        elif node.get_num_samples() < self._min_samples_split:
            return True

        return False

    def should_split(self, node, left_child, right_child):
        # compute criterion and see if it improves
        # check left_child.get_num_samples()
        pass

    def split_tree(self, node):
        # @asror i think this needs to be first  --  don't agree
        if self.should_stop(node): return
        # best threshold in any column that gives the lowest gini/mse 
        column, threshold = node.find_best_split(self._criterion)

        left_data, right_data, left_labels, right_labels = node.split_node(column, threshold)
        child_depth = node.get_depth() + 1
        left_node = Node(left_data, left_labels, parent=node, depth=child_depth)
        right_node = Node(right_data, right_labels, parent=node, depth=child_depth)
        if self.should_split(node, left_node, right_node):
            node.add_children(left_node=left_node, right_node=right_node)
            node.add_rule(column, threshold)
            self.split_tree(left_node)
            self.split_tree(right_node)


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
        if n == 0:  # don't agree - return the GINI coeff without a split
            return 0.0

        # Getting the probability to see each of the classes 
        # assumes only two classes, idk how many are in our data @asror
        p1 = left_count / n
        p2 = right_count / n
        
        # Calculating GINI -- don't think it is good
        gini = 1 - (p1 ** 2 + p2 ** 2) 
        return gini

    def compute_cross_entropy(self):
        pass

    def compute_mse(self):
        actual_values = self._labels[:, -1]
        if len(actual_values) == 0:   # empty data
            mse = 0
            
        else:
            #temporary placeholder till we implement a more robust predict method
            prediction = np.mean(actual_values) 
            mse = np.mean((actual_values - prediction) **2)
        
        return mse

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
            _, _, left, right = self.split_node(column, threshold)  # don't agree - this is O(n**2). We need O(nlogn)

            # Classification or Regression task ?
            if criterion == "classification": 
                left_counts, right_counts = Counter(left), Counter(right)
                score = node.compute_gini(left_counts, right_counts)
            if criterion == "regression": 
                score = node.compute_mse(left, right) 

            # improve our scores and move to the next possible threshold
            if score <= best_score:
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
            if score is not None and score <= best_score:  # don't agree - when can score be None. Why not np.inf ?
                best_score = score
                best_threshold = threshold
                best_column = column

        # returns the best column after iterating through possible columns
        # returns the best threshold in the best column
        return best_column, best_threshold
