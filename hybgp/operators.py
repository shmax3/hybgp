import operator
from .decorators import rename


add = rename('Add')(operator.add)
mul = rename('Mul')(operator.mul)
