import re
import itertools

import sympy as sp
import numpy as np
from deap import gp

from .generators import new_var, new_const


class PrimitiveTree(gp.PrimitiveTree):

    def __init__(self, content):
        self.parameters = {}
        super().__init__(content)

    def latex(self, rnd=False, decimals=2, deep=True):
        if rnd:
            subs = {key: np.around(val, decimals) for key, val in self.parameters.items()}
            return sp.printing.latex(sp.expand(self.__str__()).subs(subs))
        else:
            return sp.printing.latex(sp.expand(self.__str__(), deep))

    def sympy(self, subs_params=True, decimals=3):
        if subs_params:
            subs = {key: np.around(val, decimals) for key, val in self.parameters.items()}
            return sp.expand(self.__str__()).subs(subs)
        else:
            return sp.expand(self.__str__())

    def __str__(self):
        return super().__str__().replace("'", '').replace('"', '')


class PrimitiveSet(gp.PrimitiveSet):

    def __init__(self, name):
        super().__init__(name, 0)
        self.variables = []

    def addOperator(self, operator, arity, sympy_name):
        super().addPrimitive(operator, arity, sympy_name)

    def addFunction(self, func, name):
        self.context[name] = func

    def new_var(self):
        var = new_var()
        self.variables.append(var)
        return var

    @staticmethod
    def new_const():
        return new_const()

    @property
    def var_num(self):
        return len(self.variables)


def compile(expr, pset):
    code = str(expr)
    constants = set(re.findall(r"c[1-9][0-9]*", code))
    args = ",".join(itertools.chain(constants, pset.variables))
    return eval(f"lambda {args}: {code}", pset.context, {})
