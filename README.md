# Decision_Trees
## Authors: 
- Asrorbek Orzikulov
- Vincent Wilmet
- Matteo Salvalaggio
- Sofya Simakova
- Saureyj Verma

We implemented a Decision Tree from scratch. It can use either Cross Entropy or Gini for calculating impurity in 
classification tasks, and MSE or MAE for regression tasks. 

We benchmarked our model against the scikit-learn implementation (the graph below) and we got nearly the same results 
for splitting. 
![Graphic](https://github.com/Asrorbek-Orzikulov/Decision_Trees/blob/master/data/Diagram.jpeg) 

Please see our .ipynb file for more detailed explanation. We used the data found in benchmark
[diabetes dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

As you can see, our model achieved the same results as the Scikit-Learn implementation in both classification 
and regression tasks.
![Classification](https://github.com/Asrorbek-Orzikulov/Decision_Trees/blob/master/data/comparison_classification.png)
![Regression](https://github.com/Asrorbek-Orzikulov/Decision_Trees/blob/master/data/comparison_regression.png)

Please be sure to check out the file test_cases.py where we wrote many test cases for each function/attribute of our 
Decision Tree to verify its robustness.

**Complexity:** The complexity of the Scikit-Learn implementation is $O(n_{features} n_{samples}^2 \log (n_{samples}))$
as shown [here](https://scikit-learn.org/stable/modules/tree.html#complexity).
However, the complexity of our implementation is $O(n_{features} n_{samples}^3)$. We believe the difference arises from
the `split_by_column` method. Our implementation needs $n_{samples}^2)$ time to find the best split along a column,
while it should be possible to decrease the time to $O(n_{samples}^2 \log (n_{samples}))$. Had we implemented this
method more efficiently, the complexity of our trees would be the same as that of Scikit-Learn trees.


The complexity of our model is


