import pandas as pd


if __name__ == '__main__':
    s = pd.Series(list('abca'))
    s = pd.get_dummies(s)
    print(s)
