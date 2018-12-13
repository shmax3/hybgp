import functools


def scipy_minimize_adapter(optimizer):

    @functools.wraps(optimizer)
    def wrapped(*args, **kwargs):
        result = optimizer(*args, **kwargs)
        return result.x, result.fun,

    return wrapped
