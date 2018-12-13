"""The :mod:`egp` module provides the methods and classes to perform
Genetic Programming with DEAP and HybGP. It essentially contains the classes to
build a Genetic Program Tree, and the functions to evaluate it.
This module support GP with intermediate numerical optimization of float
parameters of individuals.
"""
import sys
import re
import random
import itertools
from inspect import isclass

import sympy as sp
import numpy as np
from deap import gp

from .operators import add, mul
from .types import Const, Var, Weighted, WSed


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
    args = ",".join(itertools.chain(constants, pset.arguments))
    return eval(f"lambda {args}: {code}", pset.context, {})


def full_depth_condition(height, depth, type_):
    return height < depth + {WSed: 3, Weighted: 2}.get(type_, 1)


def generate(pset, min_, max_, condition=full_depth_condition, type_=None):
    """Generate a Tree as a list of Primitives. The tree is build
    from the root to the leaves, and it stop growing when the
    condition is fulfilled.
    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree: condition(height, depth).
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              dependending on the condition function.
    """
    type_ = type_ or pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    try:
        while len(stack) != 0:
            depth, type_ = stack.pop()
            if condition(height, depth, type_):
                if issubclass(type_, WSed):
                    expr.append(gp.Primitive(add.__name__,
                                             [Weighted, Const],
                                             WSed))
                    expr.append(gp.Primitive(mul.__name__,
                                             [Const, Var],
                                             Weighted))
                    expr.append(Const())
                    expr.append(random.choice(pset.terminals[Var]))
                    expr.append(Const())
                elif issubclass(type_, Weighted):
                    expr.append(gp.Primitive(mul.__name__, [Const, Var], Weighted))
                    expr.append(Const())
                    expr.append(random.choice(pset.terminals[Var]))
                elif issubclass(type_, Const):
                    expr.append(Const())
                else:
                    term = random.choice(pset.terminals[Var])
                    if isclass(term):
                        term = term()
                    expr.append(term)
            else:
                if issubclass(type_, Const):
                    expr.append(Const())
                elif issubclass(type_, Var):
                    prim_oper = random.choice(pset.primitives[type_])
                    prim_term = random.choice(pset.terminals[type_])
                    if random.random() > .5:
                        expr.append(prim_oper)
                        for arg in reversed(prim_oper.args):
                            stack.append((depth+1, arg))
                    else:
                        expr.append(prim_term)
                else:
                    prim = random.choice(pset.primitives[type_])
                    expr.append(prim)
                    for arg in reversed(prim.args):
                        stack.append((depth+1, arg))
    except IndexError:
        _, _, traceback = sys.exc_info()
        raise IndexError("The gp.generate function tried to add "
                         "an entity of type '%s', but there is "
                         "none available." % (type_,), traceback)
    return expr
