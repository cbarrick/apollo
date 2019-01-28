'''Provides access to a subset of the Georgia Power target data.

The data must have been previously preprocessed into a CSV file named
``mb-007.{group}.log`` where ``{group}`` is an arbitrary identifier for the
data.
'''

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import xarray as xr

import apollo.storage


# Module level logger
logger = logging.getLogger(__name__)


def interval(year=None, month=None, day=None, hour=None, minute=None):
    '''Helper function to group a timeseries DataFrame by interval.

    Args:
        year (int): Group by rounding the year.
        month (int): Group by rounding the month.
        day (int): Group by rounding the day.
        hour (int): Group by rounding the hour.
        minute (int): Group by rounding the minute.

    Returns:
        Callable[[datetime.datetime or pandas.Timestamp], datetime.datetime]:
            A function that maps arbitrary datetimes to reference datetimes by
            rounding interval properties like `second` and `minute`. The input
            may be a :class:`datetime.datetime` from the standard library or a
            :class:`pandas.Timestamp`. The return value is a
            :class:`datetime.datetime`.

    Examples:
        Group a DataFrame into 15 minute blocks:
        >>> df.groupby(interval(minute=15))

    Warning:
        Getting the right arguments for the interval you want is subtle.
        Read the code to see exactly how this works.
    '''
    # Only one of the kwargs should be given. The rest are derived.
    # We use values that exceed the traditional range for the derived values.
    # This is to account for any number of "leap" issues. It has not been
    # demonstrated that this is needed, but it doesn't hurt the correctness.

    # If year is rounded, all 12 months should round down.
    if year is not None:
        assert month is None
        month = 13
    else:
        year = 1

    if month is not None:
        assert day is None
        day = 32
    else:
        month = 1

    if day is not None:
        assert hour is None
        hour = 25
    else:
        day = 1

    if hour is not None:
        assert minute is None
        minute = 70
    else:
        hour = 1

    if minute is None:
        minute = 1

    def round_down(num, divisor, origin=0):
        num -= origin
        num -= (num % divisor)
        num += origin
        return num

    def grouper(t):
        group_year = round_down(t.year, year, origin=1)
        group_month = round_down(t.month, month, origin=1)
        group_day = round_down(t.day, day, origin=1)
        group_hour = round_down(t.hour, hour)
        group_minute = round_down(t.minute, minute)
        return datetime(group_year, group_month, group_day, group_hour, group_minute,
                tzinfo=t.tzinfo)

    return grouper


def open_mb007(*cols, group=2017):
    '''Open a Georgia Power target file.

    The data must have been previously preprocessed into a CSV file named
    ``mb-007.{group}.log`` where ``{group}`` is an arbitrary identifier for the
    data.

    Arguments:
        *cols (str or int):
            The columns to read from the file. These may be names or indices.
            The reftime column (0) will always be read.
        group (Any):
            An identifier for the file to read.

    Returns:
        xarray.Dataset:
            The dataset giving the targets.
    '''
    # The data directory contains more than just the mb-007 labels.
    data_dir = apollo.storage.get('GA-POWER')
    path = data_dir / f'mb-007.{group}.log'

    # Ensure reftime is always selected and is a list.
    if 'reftime' not in cols and 0 not in cols:
        cols = ['reftime', *cols]
    else:
        cols = list(cols)

    # Load the CSV and aggregate by hour.
    df = pd.read_csv(path, usecols=cols, index_col='reftime')
    df = df.dropna()
    df.index = pd.to_datetime(df.index, infer_datetime_format=True)
    df = df.groupby(interval(hour=1)).mean()

    # For some reason, the index name isn't set by default.
    df.index.name = 'reftime'

    # In apollo, all datasets should be presented as xarray.
    df = df.to_xarray()

    return df