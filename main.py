import numpy as np
from collections import Counter


class BaseTree:
    def __init__(self, criterion, max_depth=None, min_samples_split=2):
        self._root = None
        self._X = None
        self._y = None
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._fitted = False

    def fit(self, X, y):
        assert isinstance(X, np.ndarray), "X should be a numpy array"
        assert isinstance(y, np.ndarray), "y should be a numpy array"
        assert X.shape[1] == len(y), "X and y should have the same length"
        self._root = Node(X, y)
        self._X = X
        self._y = y
        self.split_tree(self._root)
        self._fitted = True

    def predict(self, X_test):
        if not self._fitted:
            raise AttributeError("Fit method hasn't been called yet.")

        predictions = self.obtain_predictions(X_test, self._root)
        predictions = np.array(predictions)
        return predictions

    def obtain_predictions(self, X_test, node):
        left_child, right_child = node.get_children()
        if (left_child is None) and (right_child is None):  # if it's a leaf node
            prediction = node.make_prediction(self._criterion)
            predictions = [prediction] * node.get_num_samples()
            return predictions

        column, threshold = node.get_rule()
        is_less_or_equal = X_test[:, column] <= threshold
        left_data = X_test[is_less_or_equal, :]
        right_data = X_test[~is_less_or_equal, :]
        left_predictions = self.obtain_predictions(left_data, left_child)
        right_predictions = self.obtain_predictions(right_data, right_child)
        full_predictions = left_predictions + right_predictions
        return full_predictions

    def should_stop(self, node):
        # checking if we should stop growing the tree
        if self._max_depth is None:
            return False
        elif node.get_depth() >= self._max_depth:
            return True
        elif node.get_num_samples() < self._min_samples_split:
            return True
        return False

    def split_tree(self, node):
        if self.should_stop(node):
            return

        # best threshold in any column that gives the lowest gini/mse
        column, threshold = node.find_best_split(self._criterion)
        left_data, right_data, left_targets, right_targets = node.split_node_data(column, threshold)
        child_depth = node.get_depth() + 1
        left_node = Node(left_data, left_targets, depth=child_depth)
        right_node = Node(right_data, right_targets, depth=child_depth)
        node.add_children(left_node=left_node, right_node=right_node)
        node.add_rule(column, threshold)
        self.split_tree(left_node)
        self.split_tree(right_node)


class Node:
    def __init__(self, X, y, depth=0):
        self._data = X
        self._targets = y
        self._num_samples = len(self._targets)
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

    def get_targets(self):
        return self._targets

    def add_rule(self, column, threshold):
        self._column = column
        self._threshold = threshold

    def get_rule(self):
        return self._column, self._threshold

    def add_children(self, left_node, right_node):
        self._left_child = left_node
        self._right_child = right_node

    def get_children(self):
        return self._left_child, self._right_child

    def compute_gini(self, left_targets, right_targets):
        left_count_dict, right_count_dict = Counter(left_targets), Counter(right_targets)
        left_count, right_count = len(left_targets), len(right_targets)
        left_impurity = right_impurity = 1
        for class_count in left_count_dict.values():
            left_impurity -= (class_count*class_count) / (left_count * left_count)
        for class_count in right_count_dict.values():
            right_impurity -= (class_count*class_count) / (right_count * right_count)

        split_impurity = left_count*left_impurity + right_count*right_impurity
        split_impurity /= self._num_samples
        return split_impurity

    def compute_cross_entropy(self):  # TODO
        col = self._X[:, -1]  # not sure if this should be ._X or ._targets when doing Xtrain vs Xtest
        classes, class_counts = np.unique(col, return_counts = True)
        entropy_value = np.sum( [ (-class_counts[i]/np.sum(class_counts)) *  np.log2(class_counts[i]/np.sum(class_counts)) 
                                for i in range(len(classes)) ] )
        return entropy_value

    def compute_mse(self, left_targets, right_targets):
        left_count, right_count = len(left_targets), len(right_targets)
        left_mean = np.mean(left_targets)
        left_mse = np.mean((left_targets - left_mean) ** 2)
        right_mean = np.mean(right_targets)
        right_mse = np.mean((right_targets - right_mean) ** 2)
        split_mse = left_count*left_mse + right_count*right_mse
        split_mse /= self._num_samples
        return split_mse

    def compute_mae(self):  # TODO
        pass 

    def split_node_data(self, column, threshold):
        # splitting X and y according to a given column and threshold
        is_less_or_equal = self._data[:, column] <= threshold
        left_data = self._data[is_less_or_equal, :]
        right_data = self._data[~is_less_or_equal, :]
        left_targets = self._targets[is_less_or_equal]
        right_targets = self._targets[~is_less_or_equal]
        return left_data, right_data, left_targets, right_targets

    def split_by_column(self, column, criterion):
        best_score = np.inf
        best_threshold = None

        # get unique values of a column to be our thresholds
        column_values = self._data[:, column]
        thresholds = np.unique(column_values)
        max_value = np.max(thresholds)
        for threshold in thresholds:  # iterate through possible thresholds
            if threshold < max_value:
                is_small_or_eq = column_values <= threshold
                left_targets = self._targets[is_small_or_eq]
                right_targets = self._targets[~is_small_or_eq]
                if criterion == "classification":
                    score = self.compute_gini(left_targets, right_targets)
                elif criterion == "regression":
                    score = self.compute_mse(left_targets, right_targets)
                if score < best_score:
                    best_score = score
                    best_threshold = threshold

        return best_score, best_threshold

    def find_best_split(self, criterion):
        best_score = np.inf
        best_threshold = None
        best_column = None
        column_count = self._data.shape[1]
        for column in range(column_count):
            score, threshold = self.split_by_column(column, criterion)
            if score < best_score:
                best_score = score
                best_threshold = threshold
                best_column = column

        # returns the best threshold in the best column
        return best_column, best_threshold

    def make_prediction(self, criterion):
        if criterion == "classification":
            c = Counter(self._targets)
            largest_frequency = 0
            prediction = None
            for value, frequency in c.items():
                if frequency > largest_frequency:
                    largest_frequency = frequency
                    prediction = value
            return prediction
        elif criterion == "regression":
            return np.mean(self._targets)
