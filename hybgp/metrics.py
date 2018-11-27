import numpy as np


def mse(params, func, args, target):
    val = np.nan_to_num(func(*params, *args))
    return np.square(val - target).mean()
