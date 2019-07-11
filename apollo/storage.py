'''Tools for interacting with the Apollo database.

The Apollo database is a directory whose location is determined by the
``APOLLO_DATA`` environment variable, defaulting to ``/var/lib/apollo``.
'''

import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)


def get_root():
    '''Get the path to the Apollo database.

    This method reads from the ``APOLLO_DATA`` environment variable,
    defaulting to ``/var/lib/apollo``.

    Returns:
        Path:
            The location of the database.
    '''
    root = os.environ.get('APOLLO_DATA', '/var/lib/apollo')
    root = Path(root)

    if not root.is_absolute():
        logger.warning(f'APOLLO_DATA is not an absolute path: {root}')

    return root


def set_root(path):
    '''Set the path to the Apollo database.

    This method works by setting the ``APOLLO_DATA`` environment variable.

    Arguments:
        path (str or Path):
            The new location of the database.

    Returns:
        Path:
            The previous location of the database.
    '''
    old_path = get_path()
    path = Path(path).resolve()
    os.environ['APOLLO_DATA'] = str(path)
    return old_path


# This function is reexported at the top-level. Prefer ``apollo.path``.
def path(component):
    '''Get a path within the Apollo database.

    Arguments:
        component (str or Path):
            A path relative to the database root.

    Returns:
        Path:
            An absolute path within the database.
    '''
    component = Path(component)

    if component.is_absolute():
        raise ValueError('Database paths must be relative')

    path = get_root() / component
    path = path.resolve()
    return path
