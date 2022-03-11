from main import Node
import numpy as np


class TestSuite:
    """
    Create a suite of tests similar to unittest.
    """
    def __init__(self):
        """
        Creates a test suite object
        """
        self._total_tests = 0
        self._failures = 0

    def run_test(self, computed, expected, message=""):
        """
        Compare computed and expected
        If not equal, print message, computed, expected
        """
        self._total_tests += 1
        if isinstance(computed, np.ndarray):
            comparison = computed == expected
            equal_arrays = comparison.all()
            if not equal_arrays:
                msg = message + " Computed: " + str(computed)
                msg += " Expected: " + str(expected)
                print(msg)
                self._failures += 1

        elif computed != expected:
            msg = message + " Computed: " + str(computed)
            msg += " Expected: " + str(expected)
            print(msg)
            self._failures += 1

    def report_results(self):
        """
        Report back summary of successes and failures
        from run_test()
        """
        msg = "Ran " + str(self._total_tests) + " tests. "
        msg += str(self._failures) + " failures."
        print(msg)


# informal tests of functions
def run_suite():
    """
    Some informal testing code
    """
    # creating a TestSuite object
    suite = TestSuite()
    X = np.array([[3, 4], [1, 2]])
    y = np.array([6, 5])
    node = Node(X, y)
    suite.run_test(node.get_num_samples(), 2, "Test 1.1: Node init")
    suite.run_test(node.get_depth(), 0, "Test 1.2: Node init")

    # testing the split_node_data method:
    left_data, right_data, left_labels, right_labels = node.split_node_data(1, 3)
    suite.run_test(left_data, np.array([[1, 2]]), "Test 2.1: split_node_data")
    print("passed")
    suite.run_test(right_data, np.array([[3, 4]]), "Test 2.2: split_node_data")
    suite.run_test(left_labels, np.array([5]), "Test 2.3: split_node_data")
    suite.run_test(right_labels, np.array([6]), "Test 2.4: split_node_data")

    # testing the compute_gini method:
    X2 = np.array([[6.7, 3.3],
                   [6.7, 3.0],
                   [6.3, 2.5],
                   [6.5, 3.0],
                   [6.2, 3.4],
                   [5.9, 3.0],
                   [6.1, 2.8],
                   [6.4, 2.9],
                   [6.6, 3.0],
                   [6.8, 2.8]])

    y2 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    node = Node(X2, y2)
    _, _, left_targets, right_targets = node.split_node_data(0, 5.9)
    gini_imp = node.compute_gini(left_targets, right_targets)
    suite.run_test(round(gini_imp, 3), 0.444, "Test 3.1: compute_gini")
    _, _, left_targets, right_targets = node.split_node_data(0, 6.2)
    gini_imp = node.compute_gini(left_targets, right_targets)
    suite.run_test(round(gini_imp, 3), 0.476, "Test 3.2: compute_gini")
    _, _, left_targets, right_targets = node.split_node_data(0, 6.7)
    gini_imp = node.compute_gini(left_targets, right_targets)
    suite.run_test(round(gini_imp, 3), 0.4, "Test 3.3: compute_gini")
    _, _, left_targets, right_targets = node.split_node_data(1, 2.9)
    gini_imp = node.compute_gini(left_targets, right_targets)
    suite.run_test(round(gini_imp, 3), 0.317, "Test 3.4: compute_gini")

    # testing the split_by_column and find_best_split methods:
    best_score, best_threshold = node.split_by_column(0, "gini")
    suite.run_test(round(best_score, 3), 0.4, "Test 4.1: split_by_column")
    suite.run_test(best_threshold, 6.7, "Test 4.2: split_by_column")
    best_score, best_threshold = node.split_by_column(1, "gini")
    suite.run_test(round(best_score, 3), 0.317, "Test 4.3: split_by_column")
    suite.run_test(best_threshold, 2.9, "Test 4.4: split_by_column")
    best_column, best_threshold = node.find_best_split("gini")
    suite.run_test(best_column, 1, "Test 4.5: find_best_split")
    suite.run_test(best_threshold, 2.9, "Test 4.6: find_best_split")

    # reporting the results of the test
    suite.report_results()


if __name__ == '__main__':
    run_suite()
