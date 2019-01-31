''' Utility functions used throughout Apollo '''

import functools


def _is_abstract(cls):
    if not hasattr(cls, "__abstractmethods__"):
        return False  # an ordinary class
    elif len(cls.__abstractmethods__) == 0:
        return False  # a concrete implementation of an abstract class
    else:
        return True  # an abstract class


def get_concrete_subclasses(base_class):
    ''' Recursively discover non-abstract subclasses of the given base class

    Args:
        base_class (class): base class whose subclasses should be explored

    Returns:
        list: list of non-abstract subclasses of `base_class`

    '''
    subclass_models = [get_concrete_subclasses(subclass) for subclass in base_class.__subclasses__()]
    combined = functools.reduce(lambda a, b: a + b, subclass_models, [])
    if not _is_abstract(base_class):
        return [base_class] + combined
    else:
        return combined
