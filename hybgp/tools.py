import numpy as np


def train_test_split(X, y, n_split):
    size = len(y)
    block_size = size // n_split
    X_train = tuple(np.concatenate([col[block_size*i:block_size*(i+1)]
                                    for i in range(n_split)
                                    if not i % 2]) for col in X)
    X_test = tuple(np.concatenate([col[block_size*i:block_size*(i+1)]
                                   for i in range(1, n_split)
                                   if i % 2]) for col in X)
    y_train = np.concatenate([y[block_size*i:block_size*(i+1)]
                              for i in range(n_split) if not i % 2])
    y_test = np.concatenate([y[block_size*i:block_size*(i+1)]
                             for i in range(1, n_split) if i % 2])
    return X_train, X_test, y_train, y_test
