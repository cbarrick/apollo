import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)


def get_root():
    '''Set the storage root for persistent data in Apollo.

    This method reads from the ``APOLLO_DATA`` environment variable,
    defaulting to ``/var/lib/apollo``.

    Returns:
        root (Path):
            The location of the storage root.
    '''
    root = os.environ.get('APOLLO_DATA', '/var/lib/apollo')
    root = Path(root)

    if not root.is_absolute():
        logger.warning(f'storage root is not absolute: {root}')

    return root


def set_root(root):
    '''Set the storage root for persistent data in Apollo.

    This method works by setting the ``APOLLO_DATA`` environment variable.

    Arguments:
        root (str or Path):
            The new location of the storage root.

    Returns:
        old_root (Path):
            The previous storage root.
    '''
    old_root = get_root()
    root = Path(root).resolve()
    os.environ['APOLLO_DATA'] = str(root)
    return old_root


def get(component):
    '''Access a directory under the storage root.

    Arguments:
        component (str or Path):
            A path relative to the storage root.

    Returns:
        path (Path):
            An absolute path to a directory under the storage root.
            The directory is automatically created if it does not exist.
    '''
    root = get_root()
    path = Path(root) / Path(component)
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
