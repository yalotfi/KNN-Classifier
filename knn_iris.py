import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets


class scrappy_knn():
    # Basic KNN classifier
    def fit(self, X_train, y_train, k):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def euc(self, a, b):
        # Norm vectorization is more efficient
        dist = np.linalg.norm(a - b)
        return dist

    def closest(self, row, k):
        # Initialize baselines
        best_dist = self.euc(row, self.X_train[0])
        best_index = 0

        # If new distance is better, keep it
        for i in range(self.k, len(self.X_train)):
            dist = self.euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

    def predict(self, X_test, k):
        # Use distance calculations to predict class
        predictions = []
        for row in X_test:
            label = self.closest(row, k)
            predictions.append(label)
        return predictions


def process_data():
    # Load iris train and test sets
    iris = datasets.load_iris()
    X = np.array(iris.data)
    y = np.array(iris.target)

    # Split into train, dev, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    return X_train, X_test, y_train, y_test


def knn(k_nieghbor):
    # Step 1: Load data
    X_train, X_test, y_train, y_test = process_data()

    # Step 2: Build KNN Model
    classifier = scrappy_knn()
    classifier.fit(X_train, y_train, k_nieghbor)
    predictions = classifier.predict(X_test, k_nieghbor)

    # Step 3: Assess Model
    print(accuracy_score(y_test, predictions))


if __name__ == '__main__':
    # Test range of k-values, ideally avoid k = 1
    knn(1)
    knn(3)
    knn(7)
    knn(13)
