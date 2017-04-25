import numpy as np
import knn_classifier as knn
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets


def process_data():
    # Load iris train and test sets
    iris = datasets.load_iris()
    X = np.array(iris.data)
    y = np.array(iris.target)

    # Split into train, dev, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    return X_train, X_test, y_train, y_test


def run_knn(k_neighbor):
    # Step 1: Load data
    X_train, X_test, y_train, y_test = process_data()

    # Step 2: Build KNN Model
    classifier = knn.scrappy_knn()
    classifier.fit(X_train, y_train, k_neighbor)
    predictions = classifier.predict(X_test, k_neighbor)

    # Step 3: Assess Model
    print('For K = {0}: {1}%'.format(
        k_neighbor, round(accuracy_score(y_test, predictions) * 100, 2))
    )


if __name__ == '__main__':
    # Test range of k-values, ideally avoid k = 1
    run_knn(1)
    run_knn(3)
    run_knn(7)
    run_knn(13)
