import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


def normalize_df(df):
    return (df - df.mean()) / (df.max() - df.min())


def process_csv(path, colnames):
    # Import CSV
    df = pd.read_csv(
        path, encoding='utf-8', header=None, skiprows=1, names=colnames
    )

    # Subset strings and numerics
    numerics = df.select_dtypes(exclude=['object'])
    factors = df.select_dtypes(include=['object']).drop(['y'], axis=1)
    labels = df.loc[:, 'y']

    # Factorize the dataframe and series (matrix and vector)
    factors = factors.apply(lambda x: pd.factorize(x)[0])
    labels = labels.factorize()[0]

    # Concatenate factorized and numeric data subsets
    features = np.array(normalize_df(pd.concat([factors, numerics], axis=1)))
    [X_train, X_test,
     y_train, y_test] = train_test_split(features, labels, test_size=0.3)

    # Train and test sets for X and y
    return X_train, X_test, y_train, y_test


def main():
    # File path to sampled data set
    file_path = 'data/bank-sampled-cleaned.csv'
    names = [  # Save name headers because some encoding issue in raw file
        'age', 'job', 'marital',
        'education', 'default', 'housing',
        'loan', 'contact', 'month',
        'day_of_week', 'duration', 'campaign',
        'pdays', 'previous', 'poutcome',
        'emp_var_rate', 'cons_price_idx', 'cons_conf_idx',
        'euribor3m', 'nr_employed', 'y']

    # Process data with function from above
    [X_train, X_test,
     y_train, y_test] = process_csv(path=file_path, colnames=names)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # Write csv files ready for analysis
    np.savetxt('data/x_train.csv', X_train, delimiter=',')
    np.savetxt('data/X_test.csv', X_test, delimiter=',')
    np.savetxt('data/y_train.csv', y_train, delimiter=',')
    np.savetxt('data/y_test.csv', y_test, delimiter=',')


if __name__ == '__main__':
    main()
