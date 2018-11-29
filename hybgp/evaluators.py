import inspect
import numpy as np


def linear_evaluator(metric, args, optimizer, toolbox):
    def evaluate(individual):
        func = toolbox.compile(individual)
        args_sign = inspect.signature(func).parameters
        init_appr = np.array([individual.parameters.get(const, .0)
                              for const in args_sign
                              if const.startswith('c')])
        params, fitness_val, *_ = optimizer(metric, init_appr, (func, *args))
        individual.parameters = dict(zip(args_sign, params))
        return fitness_val,

    return evaluate


def tt_evaluator(metric, args_train, args_test, target_train, target_test, optimizer, toolbox):
    def evaluate(individual):
        func = toolbox.compile(individual)
        args_sign = inspect.signature(func).parameters
        init_appr = np.array([individual.parameters.get(const, .0)
                              for const in args_sign
                              if const.startswith('c')])
        params, fitness_val_train, *_ = optimizer(metric, init_appr, (func, args_train, target_train))
        individual.parameters = dict(zip(args_sign, params))
        fitness_val_test = metric(params, func, args_test, target_test)
        return fitness_val_test + 0.02 * len(args_sign),

    return evaluate
