import functools


def rename(new_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)
        wrapped.__name__ = new_name
        return wrapped
    return decorator
