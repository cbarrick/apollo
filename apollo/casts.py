'''Functions for casting between types, typically by parsing strings
'''

import time

import pandas as pd


def utc_timestamp(*args, **kwargs):
    '''Construct a UTC timestamp.

    This is equivalent to the constructor for :class:`pandas.Timestamp` except
    that the returned timestamp is always timezone-aware and set to UTC. This
    function implements the following conventions:

    - Explicit timestamps without timezones are localized to UTC.
    - Explicit timestamps with timezones are converted to UTC.
    - Implicit timestamps refer to the current time in UTC. This is equivalent
      to localizing to the current timezone then converting to UTC.

    Arguments:
        *args: Forwarded to :class:`pandas.Timestamp`.
        **kwargs: Forwarded to :class:`pandas.Timestamp`.

    Returns:
        pandas.Timestamp:
            A timestamp set to the UTC timezone.
    '''
    if 'ts_input' in kwargs:
        ts_input = kwargs['ts_input']
    elif len(args) != 0:
        ts_input = args[0]
    else:
        ts_input = None

    ts = pd.Timestamp(*args, **kwargs)

    # - When the timestamp is tz-aware, we convert it to UTC.
    # - When the timestamp is tz-naive but explict, we localize it to UTC.
    # - When the timestamp is tz-naive and 'now' or 'today', we localize it to
    #   the system timezone then convert it to UTC.
    if ts.tz is None:
        if ts_input == 'now' or ts_input == 'today':
            localtz = time.localtime().tm_gmtoff
            ts = ts.tz_localize(localtz)
            ts = ts.tz_convert('UTC')
        else:
            ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')

    return ts


def identity(x):
    '''An identity function.
    '''
    return x


def parse_bool(x):
    '''Parse a boolean.

    Strings are parsed as case-insensitive. The following inputs are
    considered false:

    - ``'false'``
    - ``'off'``
    - ``'none'``
    - ``'no'``
    - ``'n'``

    All other inputs are forwarded to :class:`bool`.

    Arguments:
        x: The thing to parse.

    Returns:
        bool: The corresponding boolean.
    '''
    if x.lower() in ['false', 'off', 'none', 'no', 'n']:
        return False
    else:
        return bool(x)


def parse_tuple(x, elm_type=identity):
    '''Parse a tuple.

    If ``x`` is a string, it is split on commas into a tuple. Otherwise ``x``
    is treated as an iterable whose elements are copied into a new tuple.

    Each element of ``x`` is parsed by the function ``elm_type``. The default
    is the identity function.

    Arguments:
        x: The thing being parsed.
        elm_type: A function to parse each element.

    Returns:
        tuple: The parsed tuple.
    '''
    if isinstance(x, str):
        return tuple(elm_type(e) for e in x.split(','))
    else:
        return tuple(elm_type(e) for e in x)
