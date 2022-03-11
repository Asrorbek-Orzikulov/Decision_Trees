from main import BaseTree, Node
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
    X = np.array([[1, 2], [3, 4]])
    y = np.array([5, 6])
    node = Node(X, y)
    suite.run_test(node.get_num_samples(), 2, "Test 1.1: Node init")
    suite.run_test(node.get_depth(), 0, "Test 1.2: Node init")

    # testing the split_node_data method:
    left_data, right_data, left_labels, right_labels = node.split_node_data(0, 1)
    suite.run_test(left_data, np.array([[1, 2]]), "Test 2.1: split_node_data")
    print("passed")
    suite.run_test(right_data, np.array([[3, 4]]), "Test 2.2: split_node_data")
    suite.run_test(left_labels, np.array([5]), "Test 2.3: split_node_data")
    suite.run_test(right_labels, np.array([6]), "Test 2.4: split_node_data")



    #     def split_node_data(self, column, threshold):

    # clf = BaseTree("classification", max_depth=1, min_samples_split=2)

    # reporting the results of the test
    suite.report_results()


if __name__ == '__main__':
    run_suite()
