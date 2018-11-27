import inspect
import functools
import numpy as np


def evaluator(metrics, kwargs_list, optimizer, toolbox):
    def evaluate(individual):
        def metric(*args):
            return sum(functools.partial(m, **k)(*args)
                       for m, k in zip(metrics, kwargs_list))

        func = toolbox.compile(individual)
        args_sign = inspect.signature(func).parameters
        init_appr = np.array([individual.parameters.get(const, .0)
                              for const in args_sign
                              if const.startswith('c')])
        params, fitness_val, *_ = optimizer(metric, init_appr, (func,))
        individual.parameters = dict(zip(args_sign, params))
        return fitness_val,

    return evaluate
