import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


def normalize_df(df):
    return (df - df.min()) / (df.max() - df.min())


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
    features = normalize_df(pd.concat([factors, numerics], axis=1))

    # Split Data
    [X_train, X_test,
     y_train, y_test] = train_test_split(
        np.array(features), labels, test_size=0.3)
    headers = features.columns.values

    # Train and test sets for X and y
    return X_train, X_test, y_train, y_test, headers


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
     y_train, y_test,
     headers] = process_csv(path=file_path, colnames=names)

    # # Append labels to train and test for export
    m_train = len(y_train)
    m_test = len(y_test)
    train = np.hstack((X_train, np.reshape(y_train, (m_train, 1))))
    test = np.hstack((X_test, np.reshape(y_test, (m_test, 1))))

    # print(X_train.shape)
    # print(train.shape)
    # print(X_test.shape)
    # print(test.shape)
    # print(headers)

    # Write csv files ready for analysis
    np.savetxt('train.csv', train, delimiter=',')
    np.savetxt('test.csv', test, delimiter=',')
    np.savetxt('headers.csv', headers, delimiter=',', fmt='%s')


if __name__ == '__main__':
    main()
