# Decision_Trees
# Authors: 
- Asrorbek Arzikulov
- Vincent Wilmet
- Matteo Salvalaggio
- Sofya Simakova
- Sauraj Verma


We implemented a Decision Tree from scratch. It can use either Cross Entropy or Gini for calulating impurity in classification tasks, and MSE or MAE for regression tasks. 

We benchmark our model against the scikit-learn implementation (graphic below) and we get nearly the same results for splitting. 
![Graphic](https://github.com/Asrorbek-Orzikulov/Decision_Trees/blob/master/data/Diagram.jpeg) 

Please see our .ipynb file for more detailed explanation. We used the data found in benchmark [diabetes dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

As you can see, our model acheives the same results as the sci-kit learn implimentation in both classification and regression tasks.
![Classification](https://github.com/Asrorbek-Orzikulov/Decision_Trees/blob/master/data/comparison_classification.jpeg)
![Regression](https://github.com/Asrorbek-Orzikulov/Decision_Trees/blob/master/data/comparison_regression.jpeg)

Please be sure to check out the file test_cases.py where we wrote many test cases for each function/attribute of our Decision Tree to verify its robsutness.  
