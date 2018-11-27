import inspect
import numpy as np


def evaluator(metric, optimizer, toolbox, args, target):

    def evaluate(individual):
        func = toolbox.compile(individual)
        args_sign = inspect.signature(func).parameters
        init_appr = np.array([individual.parameters.get(const, .0)
                              for const in args_sign
                              if const.startswith('c')])
        params, fitness_val, *_ = optimizer(metric, init_appr, (func, args, target))
        individual.parameters = dict(zip(args_sign, params))
        return fitness_val,

    return evaluate
