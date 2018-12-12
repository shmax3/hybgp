"""The :mod:`egp` module provides the methods and classes to perform
Genetic Programming with DEAP and HybGP. It essentially contains the classes to
build a Genetic Program Tree, and the functions to evaluate it.
This module support GP with intermediate numerical optimization of float
parameters of individuals.
"""
import re
import itertools

import sympy as sp
import numpy as np
from deap import gp

from .generators import new_var, new_const


class PrimitiveTree(gp.PrimitiveTree):
    """Tree specifically formatted for optimization of genetic
    programming operations with intermediate optimization of float
    parameters. The tree is represented with a
    list where the nodes are appended in a depth-first order.
    The nodes appended to the tree are required to
    have an attribute *arity* which defines the arity of the
    primitive. An arity of 0 is expected from terminals nodes.
    It can produce sympy and latex expression corresponding to
    individual.
    """
    def __init__(self, content):
        self.parameters = {}
        super().__init__(content)

    def latex(self, subs_params=True, decimals=3):
        """It constructs a latex expression of the individual.

        :param subs_params: If it is True it returns formula with substituted
        parameters else parameters are kept symbolical.
        :param decimals: Number of decimals displayed if subs_params is True.
        :return: str in LaTeX notation.
        """
        return sp.printing.latex(self.sympy(subs_params, decimals))

    def sympy(self, subs_params=True, decimals=3):
        """It constructs a sympy expression of the individual.

        :param subs_params: If it is True it returns formula with substituted
        parameters else parameters are kept symbolical.
        :param decimals: Number of decimals displayed if subs_params is True.
        :return: sympy expression.
        """
        if subs_params:
            subs = {key: np.around(val, decimals)
                    for key, val in self.parameters.items()}
            return sp.expand(self.__str__()).subs(subs)
        else:
            return sp.expand(self.__str__())


def compile(expr, pset):
    """Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    code = str(expr)
    constants = set(re.findall(r"c[1-9][0-9]*", code))
    args = ",".join(itertools.chain(constants, pset.variables))
    return eval(f"lambda {args}: {code}", pset.context, {})
