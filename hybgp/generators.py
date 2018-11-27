def new_const_gen():
    i = 1
    while True:
        yield f'c{i}'
        i += 1


new_const_creator = new_const_gen()


def new_const():
    return next(new_const_creator)


def new_var_gen():
    i = 1
    while True:
        yield f'x{i}'
        i += 1


new_var_creator = new_var_gen()


def new_var():
    return next(new_var_creator)
