# KNN Classifier from Scratch

Implement a K-Nearest Neighbor classifier from scratch on the famous iris-flower dataset. The algorithm determines the value of an unseen data point based on the value of its neighbors. It calculates euclidean distance, a beefed up Pythagorean Theorem, to determine the distance between a new point and the closest data. Depending on how many points are in the vicinity, it classifies the new data point accordingly. KNN can handle binary classification and beyond to "n" number of classes.

### Pros and Cons of KNN

KNN Classifier's best attribute is its relative simplicity. It is not difficult to implement and its intuition is very basic as well. KNN's greatest drawback is computational inefficiency. The algorithm iterates over every data point, making implementation over a large dataset cumbersome. One could reduce this problem by vectorizing the implementation, which is easy to do in MATLAB or Octave, but requires more dependencies in Python or C/C++. Another drawback is that it becomes very difficult to represent features beyond 2 or 3 dimensions and its decision boundaries, making KNN a black-box of sorts.

# Dependencies:

* `train_test_split` from `sklearn.cross_validation`

* `accuracy_score` from `sklearn.metrics`

* `numpy` to calculate distance between to points

* `datasets` from `sklearn`

* `datasets.load_iris()`

# Results:

Testing a range of k values, the model consistently scores above 92% but varies around an average of 94%.  The best score was 97% with 'k = 7', meaning a new data point is compared against its 7 nearest neighbors. This could be tweaked further, but I am happy with these results. Any further improvement would be some visualization of the model, but that can be tricky with high dimensional data.  