import numpy as np
import knn_iris as knn
import process_csv as pr


def main():
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
    print(np.array(features).shape, type(labels))
    print(np.array(features)[0, 0:3])

    # Build classifier
    classifier = knn.scrappy_knn()
    print(type(classifier))


if __name__ == '__main__':
    main()
