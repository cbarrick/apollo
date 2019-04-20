'''Utilities for working with Timestamps.
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
