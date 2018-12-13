def new_const_gen():
    i = 1
    while True:
        yield f'c{i}'
        i += 1


c = new_const_gen()


class Const:

    __slots__ = ('name', 'arity', 'args', 'ret', 'seq')

    def __init__(self, name=None):
        self.arity = 0
        self.ret = Const
        self.args = []
        if name is None:
            self.name = next(c)
        else:
            self.name = name

    def __str__(self):
        return str(self.name)

    def format(self, *args, **kwargs):
        return self.__str__().format(*args, **kwargs)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot)
                       for slot in self.__slots__)
        else:
            return NotImplemented


class Var:
    pass


class Weighted:
    pass


class WSed:
    pass
