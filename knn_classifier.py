import numpy as np


class scrappy_knn():
    # Basic KNN classifier
    def fit(self, X_train, y_train, k):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def euc(self, a, b):
        # Norm difference is more efficient
        return np.linalg.norm(a - b)

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
