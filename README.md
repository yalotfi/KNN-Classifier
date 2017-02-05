# KNN Classifier from Scratch

Implement a K-Nearest Neighbor classifier from scratch on the famous iris-flower dataset. The algorithm determines the value of an unseen data point based on the value of its neighbors. It calculates euclidean distance, a beefed up Pythagorean Theorem, to determine the distance between a new point and the closest data. Depending on how many points are in the vicinity, it classifies the new data point accordingly. KNN can handle binary classification and beyond to "n" number of classes.

### Dependencies:

* `train_test_split` from `sklearn.cross_validation`

* `accuracy_score` from `sklearn.metrics`

* `distance` method from `scipy.spatial`

* `datasets` from `sklearn`

* `datasets.load_iris()`