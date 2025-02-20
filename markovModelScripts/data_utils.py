import warnings

def split_data(df, train_size=0.8, embargo_size=0.01):
    n = len(df)
    train_end = int(n * train_size)
    embargo_end = train_end + int(n * embargo_size)
    train = df[:train_end]
    embargo = df[train_end:embargo_end]
    test = df[embargo_end:]
    if len(embargo) < 5:
        warnings.warn("Embargo period is less than 5 data points")
    return train, embargo, test