import numpy as np
from collections import Counter


class DecisionTree:
    def __init__(self, criterion, max_depth=None, min_samples_split=2):
        ''' Initialization of parameters:
        criterion: {gini, entropy, mse, mae}
        max_depth: defaults to None (runs until base case is met)
        min_samples_split: defaults to 2, meaning we need at least enough to make a Left and Right child'''
        self._root = None
        self._criterion = criterion
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._fitted = False

    def fit(self, X, y):
        ''' fit method 
        Assumes X and y are type np.ndarray'''

        #verify inputs
        assert isinstance(X, np.ndarray), "X should be a numpy array"
        assert isinstance(y, np.ndarray), "y should be a numpy array"
        assert X.shape[0] == len(y), "X and y should have the same length"

        self._root = Node(X, y)
        self.split_tree(self._root)
        self._fitted = True # allows us to procceed in predict()

    def predict(self, X_test):
        ''' predict method 
        returns a np array of every sample in every node's cassification '''
        if not self._fitted: # verify that the tree had been constructed 
            raise AttributeError("Fit method hasn't been called yet.")

        predictions = []
        row_count = X_test.shape[0]
        for row_idx in range(row_count):
            prediction = self.obtain_predictions(X_test[row_idx, :], self._root)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions

    def obtain_predictions(self, test_row, node):
        ''' called by predict
        recurses until base case (left and right child are none) then makes a prediction'''
        left_child, right_child = node.get_children()
        if (left_child is None) and (right_child is None):  # if it's a leaf node
            prediction = node.make_prediction(self._criterion)
            return prediction

        column, threshold = node.get_rule()
        if test_row[column] <= threshold:
            return self.obtain_predictions(test_row, left_child)
        else:
            return self.obtain_predictions(test_row, right_child)

    def should_stop(self, node):
        ''' checking if we should stop growing the tree'''
        if self._max_depth is None:
            return False
        elif node.get_depth() >= self._max_depth:
            return True
        elif node.get_num_samples() < self._min_samples_split:
            return True
        elif len(np.unique(node.get_targets())) == 1:
            return True
        return False # all good, carry on

    def split_tree(self, node):
        ''' splits current node into a left and right tree 
        assigns the children and the rule to their parent '''
        if self.should_stop(node):
            return # verify we aren't at a base case

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
        ''' Initialization
        X: np.ndarray
        y: np.ndarray
        depth = current depth. defaults to 0'''
        self._data = X
        self._targets = y
        self._num_samples = len(self._targets)
        self._depth = depth
        self._column = None
        self._threshold = None
        self._left_child = None
        self._right_child = None 

    def get_depth(self):
        ''' returns this node's depth'''
        return self._depth

    def get_num_samples(self):
        ''' returns the number of data points in y'''
        return self._num_samples

    def get_data(self):
        ''' returns X'''
        return self._data

    def get_targets(self):
        ''' returns y'''
        return self._targets

    def add_rule(self, column, threshold):
        ''' assigns the column and threshold used for splitting this node into its children '''
        self._column = column
        self._threshold = threshold

    def get_rule(self):
        ''' returns the column and threshold used for splitting this node into its children'''
        return self._column, self._threshold

    def add_children(self, left_node, right_node):
        ''' assigns the children that were created using the column and threshold of splitting '''
        self._left_child = left_node
        self._right_child = right_node

    def get_children(self):
        ''' returns the children that were created using the column and threshold of splitting'''
        return self._left_child, self._right_child

    def compute_gini(self, left_targets, right_targets):
        ''' returns the weighted gini impurity of the proposed left and right split'''
        left_count_dict, right_count_dict = Counter(left_targets), Counter(right_targets)
        left_count, right_count = len(left_targets), len(right_targets)
        left_impurity, right_impurity = 1, 1

        for class_count in left_count_dict.values():
            left_impurity -= (class_count*class_count) / (left_count * left_count)
        for class_count in right_count_dict.values():
            right_impurity -= (class_count*class_count) / (right_count * right_count)

        split_impurity = left_count*left_impurity + right_count*right_impurity # weighted
        split_impurity /= self._num_samples # scale
        return split_impurity

    def compute_cross_entropy(self, left_targets, right_targets):
        ''' returns the weighted cross entropy of the proposed left and right split'''
        left_count_dict, right_count_dict = Counter(left_targets), Counter(right_targets)
        left_count, right_count = len(left_targets), len(right_targets)
        left_entropy = sum((-class_count/left_count) * np.log2(class_count/left_count)
                           for class_count in left_count_dict.values())
        right_entropy = sum((-class_count/right_count) * np.log2(class_count/right_count)
                            for class_count in right_count_dict.values())
        split_impurity = left_count*left_entropy + right_count*right_entropy
        split_impurity /= self._num_samples
        return split_impurity

    def compute_mse(self, left_targets, right_targets):
        ''' returns the weighted MSE of the proposed left and right split'''
        left_count, right_count = len(left_targets), len(right_targets)
        left_mean = np.mean(left_targets)
        left_mse = np.mean((left_targets - left_mean) ** 2)
        right_mean = np.mean(right_targets)
        right_mse = np.mean((right_targets - right_mean) ** 2)
        split_mse = left_count*left_mse + right_count*right_mse
        split_mse /= self._num_samples
        return split_mse

    def compute_mae(self, left_targets, right_targets):
        ''' returns the weighted MAE of the proposed left and right split'''
        left_count, right_count = len(left_targets), len(right_targets)
        left_median = np.median(left_targets)
        left_mae = np.mean((left_targets - left_median) ** 2)
        right_median = np.median(right_targets)
        right_mae = np.mean((right_targets - right_median) ** 2)
        split_mae = left_count*left_mae + right_count*right_mae
        split_mae /= self._num_samples
        return split_mae

    def split_node_data(self, column, threshold):
        ''' returns the proposed split for X and y according to a given column and threshold '''
        is_less_or_equal = self._data[:, column] <= threshold
        left_data = self._data[is_less_or_equal, :]
        right_data = self._data[~is_less_or_equal, :]
        left_targets = self._targets[is_less_or_equal]
        right_targets = self._targets[~is_less_or_equal]
        return left_data, right_data, left_targets, right_targets

    def split_by_column(self, column, criterion):
        ''' returns the best threshold given the column and criterion used to determine a threshold'''
        best_score = np.inf
        best_threshold = None

        # get unique values of a column to be our thresholds
        column_values = self._data[:, column]
        max_value = np.max(column_values)
        for threshold in column_values:
            if threshold < max_value:
                is_small_or_eq = column_values <= threshold
                left_targets = self._targets[is_small_or_eq]
                right_targets = self._targets[~is_small_or_eq]
                if criterion == "gini":
                    score = self.compute_gini(left_targets, right_targets)
                elif criterion == "entropy":
                    score = self.compute_cross_entropy(left_targets, right_targets)
                elif criterion == "mse":
                    score = self.compute_mse(left_targets, right_targets)
                elif criterion == "mae":
                    score = self.compute_mae(left_targets, right_targets)

                if score < best_score:
                    best_score = score
                    best_threshold = threshold

        return best_score, best_threshold

    def find_best_split(self, criterion):
        ''' returns best column and threshold after having iterated over all columns'''
        best_score = np.inf
        best_threshold = None
        best_column = None
        column_count = self._data.shape[1]
        for column in range(column_count): #looking for the best column
            score, threshold = self.split_by_column(column, criterion)
            if score < best_score:
                best_score = score
                best_threshold = threshold
                best_column = column

        # returns the best threshold in the best column
        return best_column, best_threshold

    def make_prediction(self, criterion):
        ''' returns the classified label or MSE/MAE value'''
        if (criterion == "gini") or (criterion == "entropy"):
            c = Counter(self._targets)
            largest_frequency = 0
            prediction = None
            for value, frequency in c.items():
                if frequency > largest_frequency:
                    largest_frequency = frequency
                    prediction = value
            return prediction
        elif criterion == "mse":
            return np.mean(self._targets)
        elif criterion == "mae":
            return np.median(self._targets)
