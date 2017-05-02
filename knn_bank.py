import numpy as np
import knn_classifier as knn
import process_csv as pr
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


def main(k):
    # Data Dependecies
    file_path = 'data/bank-sampled-cleaned.csv'
    names = [  # Save feature names
        'age', 'job', 'marital',
        'education', 'default', 'housing',
        'loan', 'contact', 'month',
        'day_of_week', 'duration', 'campaign',
        'pdays', 'previous', 'poutcome',
        'emp_var_rate', 'cons_price_idx', 'cons_conf_idx',
        'euribor3m', 'nr_employed', 'y']

    # Import Data
    features, labels = pr.process_csv(path=file_path, colnames=names)
    [X_train, X_test,
     y_train, y_test] = train_test_split(
        np.array(features), labels, test_size=0.3, random_state=99)

    # Build classifier
    classifier = knn.scrappy_knn()
    classifier.fit(X_train, y_train, k)
    predictions = classifier.predict(X_test, k)

    # Step 3: Assess Model
    print('For K = {0}: {1}%'.format(
        k, round(accuracy_score(y_test, predictions) * 100, 2))
    )


if __name__ == '__main__':
    for k in range(3, 11, 2):
        main(k)
