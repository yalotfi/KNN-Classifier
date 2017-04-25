import pandas as pd


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
    features = pd.concat([factors, numerics], axis=1)

    # Return normalized feature matrix and binary label classes
    return normalize_df(features), pd.Series(labels)


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
    features, labels = process_csv(path=file_path, colnames=names)

    # Write csv files ready for analysis
    features.to_csv('data/bank-sample-factorized.csv')
    labels.to_csv('data/bank-sample-factorized-labels.csv')


if __name__ == '__main__':
    main()
