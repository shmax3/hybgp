import numpy as np
import inspect


def mse(weight):
    def weight_mse(params, func, args, target):
        val = np.nan_to_num(func(*params, *args))
        return weight * np.square(val - target).mean()
    return weight_mse


def parameters_num(weight):
    def par_num(params, func):
        return weight * len(inspect.signature(func).parameters)
    return par_num
