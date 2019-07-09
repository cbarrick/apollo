'''Functions for casting between types, typically by parsing strings
'''
import time

import pandas as pd

#: The timezone of the system clock.
LOCAL_TZ = time.localtime().tm_gmtoff


class Timestamp(pd.Timestamp):
    '''Apollo's timestamp type.

    Apollo timestamps extend :class:``pandas.Timestamp`` to ensure the structure
    is always timezone aware and localized to UTC.

    When constructing an Apollo timestamp, if the timezone would be ambiguous,
    it is assumed to be UTC.
    '''
    def __new__(cls, *args, **kwargs):
        if 'ts_input' in kwargs:
            ts_input = kwargs['ts_input']
        elif len(args) != 0:
            ts_input = args[0]
        else:
            ts_input = None

        ts = super().__new__(cls, *args, **kwargs)

        # - When the timestamp is tz-aware, we convert it to UTC.
        # - When the timestamp is tz-naive but explict, we localize it to UTC.
        # - When the timestamp is tz-naive and 'now' or 'today', it must be
        #   localized to the system timezone then converted to UTC.
        if ts.tz is None:
            if ts_input == 'now' or ts_input == 'today':
                ts = ts.tz_localize(LOCAL_TZ).tz_convert('UTC')
            else:
                ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')

        return ts
