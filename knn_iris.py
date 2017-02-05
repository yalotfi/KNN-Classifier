from scipy.spatial import distance
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets


class scrappy_knn():
    def fit(self, X_train, y_train, k):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def closest(self, row, k):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(self.k, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row, 1)
            predictions.append(label)
        return predictions


def euc(a, b):
    return distance.euclidean(a, b)


def process_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    return X_train, X_test, y_train, y_test


def knn(k_nieghbor):
    X_train, X_test, y_train, y_test = process_data()
    classifier = scrappy_knn()
    classifier.fit(X_train, y_train, k_nieghbor)
    predictions = classifier.predict(X_test)
    print(accuracy_score(y_test, predictions))


if __name__ == '__main__':
    knn(7)
