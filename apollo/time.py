'''Time related functionality.

These functions and classes are reexported at the top-level, e.g. prefer
``apollo.Timestamp`` to ``apollo.timestamps.Timestamp``.
'''

import time
import pandas as pd


def localtz():
    '''Return the local timezone as an offset in seconds.
    '''
    return time.localtime().tm_gmtoff


class Timestamp(pd.Timestamp):
    '''Apollo's timestamp type.

    Apollo timestamps extend :class:``pandas.Timestamp`` to ensure the structure
    is always timezone aware and localized to UTC.

    When constructing an Apollo timestamp, if the timezone would be ambiguous,
    it is assumed to be UTC. Otherwise it is converted to UTC.
    '''
    def __new__(cls, *args, **kwargs):
        # The important argument is ``ts_input``. The pandas constructor has
        # complex argument handling, so this must be as generic as possible.
        if len(args) != 0:
            ts_input = args[0]
        else:
            ts_input = kwargs.get('ts_input')

        # Delegate to pandas.
        ts = super().__new__(cls, *args, **kwargs)

        # Ensure the timezone is UTC.
        # 'now' and 'today' must first be localized to the system clock.
        if ts_input in ['now', 'today']:
            ts = ts.tz_localize(localtz()).tz_convert('UTC')
        elif ts.tz is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')

        # Ensure the result is an instance of this class.
        ts = super().__new__(cls, ts)
        return ts


class DatetimeIndex(pd.DatetimeIndex):
    '''Apollo's index type.

    Apollo timestamps extend :class:``pandas.DatetimeIndex`` to ensure the
    index is always timezone aware and localized to UTC.

    When constructing an Apollo index, if the timezone would be ambiguous, it
    is assumed to be UTC. Otherwise it is converted to UTC.

    When creating a DatetimeIndex from strings, the special strings ``'now'``
    and ``'today'`` will almost certainly be misinterpreted. Always use proper
    timestamps. The :class:`apollo.Timestamp` class will properly interpret
    the special strings.
    '''
    def __new__(cls, *args, **kwargs):
        # Delegate to pandas.
        index = super().__new__(cls, *args, **kwargs)

        # Ensure the timezone is UTC.
        if index.tz is None:
            index = index.tz_localize('UTC')
        else:
            index = index.tz_convert('UTC')

        # Ensure the result is an instance of this class.
        index = super().__new__(cls, index)
        return index


def date_range(start=None, end=None, *args, **kwargs):
    '''Like :func:`pandas.date_range` but returns :class:`apollo.DatetimeIndex`.

    When constructing an Apollo index, if the timezone would be ambiguous, it
    is assumed to be UTC. Otherwise it is converted to UTC.

    Unlike :func:`pandas.date_range`, the ``tz`` argument may only be ``None``
    or ``'UTC'``.
    '''
    # Handle start and end with the same logic as `apollo.Timestamp`.
    # This handles the strings 'now' and 'today' and enables mixed timezones.
    if start is not None: start = Timestamp(start)
    if end is not None: end = Timestamp(end)

    # Delegate to pandas, then ensure the index is an `apollo.DatetimeIndex`.
    index = pd.date_range(start, end, *args, **kwargs)
    index = DatetimeIndex(index)
    return index


def time_of_day(times):
    '''Compute time-of-day features.

    Arguments:
        times (numpy.ndarray like):
            A series of timestamps.

    Returns:
        pandas.DataFrame:
            A data frame with 2 data variables:
                - ``time_of_day_sin`` for the sin component of time-of-day
                - ``time_of_day_cos`` for the cosin component of time-of-day

            The data frame is indexed by the input timestamps.
    '''
    nanos = np.asarray(times, dtype='datetime64[ns]').astype('float32')
    seconds = nanos / 1e9
    days = seconds / 86400

    return pd.DataFrame({
        'time_of_day_sin': np.sin(days * 2 * np.pi),
        'time_of_day_cos': np.cos(days * 2 * np.pi),
    }, index=DatetimeIndex(times))


def time_of_year(times):
    '''Compute time-of-year features.

    Arguments:
        times (numpy.ndarray like):
            A series of timestamps.

    Returns:
        pandas.DataFrame:
            A data frame with 2 data variables:
                - ``time_of_year_sin`` for the sin component of time-of-year.
                - ``time_of_year_cos`` for the cosin component of time-of-year.

            The data frame is indexed by the input timestamps.
    '''
    nanos = np.asarray(times, dtype='datetime64[ns]').astype('float32')
    seconds = nanos / 1e9
    days = seconds / 86400

    # Convert the data into day and year units.
    # The time-of-year is measured in Julian years, 365.25 days of 86400 seconds
    # each. These are not exactly equal to a Gregorian year (i.e. a year on the
    # western calendar) once you get into the crazy leap rules.
    years = days / 365.25

    return pd.DataFrame({
        'time_of_year_sin': np.sin(years * 2 * np.pi),
        'time_of_year_cos': np.cos(years * 2 * np.pi),
    }, index=DatetimeIndex(times))


def is_daylight(times, lat, lon):
    '''Determine if the sun is above the horizon for the given times.

    The resulting series has a one hour leeway for both sunrise and sunset.

    Arguments:
        times (numpy.ndarray like):
            A series of timestamps.
        lat (float):
            The latitude.
        lon (float):
            The longitude.

    Returns:
        pandas.Series:
            A boolean series indexed by the times.
    '''
    times = DatetimeIndex(times, name='time')

    # Compute the time of the next sunrise, sunset, and transit (solar noon).
    # This computation assumes an altitude of sea-level and determined positions
    # based on the center of the sun. It could be adjusted for greater accuracy,
    # but the loss of generality probably isn't worth it.
    # (`rst` abreviates `rise_set_transit`)
    rst = solarposition.sun_rise_set_transit_ephem(times, lat, lon)

    # Daylight is when the next sunset preceeds the next sunrise.
    daylight = (rst.sunset < rst.sunrise)

    # Give ourselves leeway to account for the hour in which sunrise/set occurs.
    one_hour = pd.Timedelta(1, 'h')
    sunrise_edge = (rst.sunrise - rst.index < one_hour)
    sunset_edge = (rst.sunset - rst.index < one_hour)
    daylight |= sunrise_edge
    daylight |= sunset_edge

    # Ensure the series has a name.
    daylight.name = 'daylight'
    return daylight
