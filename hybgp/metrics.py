import numpy as np
import inspect


def mse(params, func, args, target):
    val = np.nan_to_num(func(*params, *args))
    return np.square(val - target).mean()


def parameters_num(params, func):
    return len(inspect.signature(func).parameters)
