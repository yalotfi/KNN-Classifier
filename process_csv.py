import pandas as pd
import numpy as np


def process_csv(path, colnames):
    # Import CSV
    df = pd.read_csv(
        path, encoding='utf-8', header=None, skiprows=1, names=colnames
    )

    # Subset strings and numerics
    numerics = df.select_dtypes(exclude=['object'])
    factors = df.select_dtypes(include=['object'])

    # Factorize the strings
    factors = factors.apply(lambda x: pd.factorize(x)[0])

    # Concatenate factorized and numeric data subsets
    clean_df = pd.concat([factors, numerics], axis=1)
    return df, clean_df


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
    raw_df, clean_df = process_csv(path=file_path, colnames=names)

    # Check that there are no missing values
    print(clean_df.apply(lambda x: np.sum(pd.isnull(x))))

    # Write a csv file ready for analysis
    clean_df.to_csv('data/bank-sample-factorized.csv')


if __name__ == '__main__':
    main()
