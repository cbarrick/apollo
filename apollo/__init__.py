from .storage import *
from .time import *


def _import_from_str(dotted):
    '''Import an object from a dotted import path.

    Arguments:
        dotted (str):
            A dotted import path. This string must contain at least one dot.
            Everything to the left of the last dot is interpreted as the import
            path to a Python module. The piece to the right of the last dot is
            the name of an object within that module.

    Returns:
        object:
            The object named by the dotted import path.
    '''
    import importlib
    (module, obj) = dotted.rsplit('.', 1)
    module = importlib.import_module(module)
    return getattr(module, obj)
