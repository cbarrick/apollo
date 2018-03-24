from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

import xarray as xr


# Names of the columns in the MB007 labels.
MB007_NAMES = (
    # Index column.
    # The name 'reftime' joins with the NAM data.
    'reftime',

    # The next three columns are undocumented.
    'unknown1', 'unknown2', 'unknown3',

    # Array A
    'UGA-A-POA-1-IRR',    # POA Irradiance
    'UGA-A-POA-2-IRR',    # POA Irradiance
    'UGA-A-POA-3-IRR',    # POA Irradiance
    'UGA-A-POA-REF-IRR',  # Cell Temp and Irradiance

    # Array B
    'UGA-B-POA-1-IRR',    # POA Irradiance
    'UGA-B-POA-2-IRR',    # POA Irradiance
    'UGA-B-POA-3-IRR',    # POA Irradiance
    'UGA-B-POA-REF-IRR',  # Cell Temp and Irradiance

    # Array D
    # NOTE: array D comes before array C
    'UGA-D-POA-1-IRR',    # POA Irradiance
    'UGA-D-POA-2-IRR',    # POA Irradiance
    'UGA-D-POA-3-IRR',    # POA Irradiance
    'UGA-D-POA-REF-IRR',  # Cell Temp and Irradiance

    # Array C
    # NOTE: array C comes after array D
    'UGA-C-POA-1-IRR',    # POA Irradiance
    'UGA-C-POA-2-IRR',    # POA Irradiance
    'UGA-C-POA-3-IRR',    # POA Irradiance
    'UGA-C-POA-REF-IRR',  # Cell Temp and Irradiance

    # Array E
    'UGA-E-POA-1-IRR',    # POA Irradiance
    'UGA-E-POA-2-IRR',    # POA Irradiance
    'UGA-E-POA-3-IRR',    # POA Irradiance
    'UGA-E-POA-REF-IRR',  # Cell Temp and Irradiance

    # MDAS weather station
    'UGA-MET01-POA-1-IRR',  # GHI
    'UGA-MET01-POA-2-IRR',  # GHI

    # SOLYS2
    'UGA-MET02-GHI-IRR',  # GHI
    'UGA-MET02-DHI-IRR',  # DHI
    'UGA-MET02-FIR-IRR',  # DLWIR
    'UGA-MET02-DNI-IRR',  # DNI
)


def interval(year=None, month=None, day=None, hour=None, minute=None):
    '''Group a timeseries DataFrame by interval.

    Example:
        Group a DataFrame into 15 minute blocks:
        >>> df.groupby(interval(minute=15))

    Gotchas:
        Getting the right arguments for the interval you want is subtle.
        Read the code to see exactly how this works.

    Args:
        year (int): Group by rounding the year.
        month (int): Group by rounding the month.
        day (int): Group by rounding the day.
        hour (int): Group by rounding the hour.
        minute (int): Group by rounding the minute.

    Returns:
        A function that maps arbitrary datetimes to reference datetimes by
        rounding interval properties like `second` and `minute`.
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


def open_mb007(*cols, data_dir='./data/GA-POWER'):
    # The data directory contains more than just the mb-007 labels.
    data_dir = Path(data_dir)
    paths = data_dir.glob('raw/mb-007.*.log.gz')

    # All columns must be given by name.
    # This enforces good style.
    if len(cols) is 0:
        cols = list(MB007_NAMES)
    for c in cols:
        assert isinstance(c, str)

    # Ensure reftime is always selected.
    if 'reftime' not in cols:
        cols = ('reftime', *cols)

    # Read each log into a dataframe.
    frames = []
    for path in paths:
        try:
            frames.append(
                pd.read_csv(
                    path, header=None, index_col='reftime', usecols=cols,
                    parse_dates=['reftime'], names=MB007_NAMES))
        except EmptyDataError:
            continue

    # Combine the dataframes and aggregate by hour.
    df = pd.concat(frames)
    df = df.dropna()
    df = df.sort_index()
    df = df.groupby(interval(hour=1)).mean()

    # For some reason, the index name isn't set by default.
    df.index.name = 'reftime'

    # For this project, all datasets should be presented as xarray.
    df = df.to_xarray()

    return df
