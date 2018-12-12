import numpy as np
import deap


class HallOfFame(deap.tools.HallOfFame):

    def insert(self, item):
        print(str(item))
        super().insert(item)


def train_test_split(X, y, n_split):
    """It splits a sample to train and test parts.

    :param X: A tuple of columns with same length.
    :param y: A column with length which equals X[0] length.
    :param n_split: Number of same sections for train-test spliting. This
    number must be greater than 1 and less than len(y).

    >>> train_test_split((np.array([1, 2]),), np.array([3, 4]), 2)
    ((np.array([1], ), (np.array([2]),) np.array([3]), np.array([4]))
    >>> train_test_split((np.array([1, 2, 3]),), np.array([4, 5, 6]), 3)
    ((np.array([1, 3], ), (np.array([2]),) np.array([4, 6]), np.array([5]))

    :return: tuple with (X_train, X_test, y_train, y_test)
    """
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
