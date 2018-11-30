import numpy as np
import inspect


def mse(params, func, args, target):
    """Mean squared error. sum((func(*params, *args) - target) ** 2) / N

    :param params: A tuple of parameters of *func* function.
    :param func: A function with prototype func(*params, *args).
    :param args: A tuple of the other arguments of *func* function.
    :param target: target - func(*params, *args) must be correct expression.
    :return: Float value of mean squared error.
    """
    val = np.nan_to_num(func(*params, *args))
    return np.square(val - target).mean()


def parameters_num(params, func):
    """Calculate the number of parameters of *func* function

    :param params: A tuple of parameters of *func* function.
    :param func: A function with prototype func(*params).
    :return:  int number of parameters of *func* function
    """
    return len(inspect.signature(func).parameters)
