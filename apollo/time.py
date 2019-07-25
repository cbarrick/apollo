'''Time related functionality.

The contents of this module are reexported at the top-level, e.g. prefer
``apollo.Timestamp`` over ``apollo.timestamps.Timestamp``.
'''

import time

import pandas as pd

import pvlib
from pvlib import solarposition


def _localtz():
    '''Return the local timezone as an offset in seconds.
    '''
    return time.localtime().tm_gmtoff


class Timestamp(pd.Timestamp):
    '''Apollo's timestamp type, extending Pandas.

    Timestamps in Apollo adhere to the following conventions:

    - Timestamps are always UTC.
    - Timezone-naive inputs are interpreted as UTC.
    - Timezone-aware inputs in a different timezone are converted to UTC.
    '''
    def __new__(cls, ts_input, tz=None):
        '''Construct a new timestamp.

        Arguments:
            ts_input (str or datetime-like):
                The raw timestamp. This is typically an ISO 8601 string or
                one of the special strings ``'now'`` and ``'today'``. Unlike
                Pandas, the special strings are considered timezone aware and
                localized to the system clock.
            tz (str or pytz.timezone or None):
                If ``ts_input`` has an ambiguous timezone, interpret it with
                this timezone. This argument cannot be given if ``ts_input``
                is timezone aware.
        '''
        # Delegate to Pandas.
        ts = pd.Timestamp(ts_input, tz=tz)

        # Ensure the timezone is UTC.
        # 'now' and 'today' must be localized to the system clock.
        if ts_input in ['now', 'today']:
            assert tz is None, 'Cannot pass tz when ts_input is "now" or "today"'
            tz = _localtz()

        if ts.tz is None:
            tz = tz or 'UTC'
            ts = ts.tz_localize(tz).tz_convert('UTC')
        else:
            assert tz is None, 'Cannot pass tz when ts_input is timezone aware'
            ts = ts.tz_convert('UTC')

        # Ensure the result is an instance of this class.
        ts = super().__new__(cls, ts)
        return ts

    def __init__(self, *args, **kwargs):
        '''
        '''
        # This method exists purely for documentation.
        return super().__init__(self, *args, **kwargs)


class DatetimeIndex(pd.DatetimeIndex):
    '''Apollo's index type, extending Pandas.

    Timestamps in Apollo adhere to the following conventions:

    - Timestamps are always UTC.
    - Timezone-naive inputs are interpreted as UTC.
    - Timezone-aware inputs in a different timezone are converted to UTC.

    When creating a DatetimeIndex from strings, the special strings ``'now'``
    and ``'today'`` will almost certainly be misinterpreted. Always use true
    timestamps. The :class:`apollo.Timestamp` class will properly interpret
    the special strings.
    '''
    def __new__(cls, data=None, tz=None, ambiguous='raise', name=None):
        '''Construct a new datetime index.

        This constructor only deals with explicit lists of timestamps. To
        generate a datetime index over a range of dates, see
        :func:`apollo.date_range`.

        Arguments:
            data (array or None):
                The list of timestamps for the index.
            tz (str or pytz.timezone or None):
                If ``data`` has an ambiguous timezone, interpret it with
                this timezone. This argument cannot be given if ``data``
                is timezone aware.
            ambiguous ('raise' or 'infer' or 'NaT' or array of bool):
                Timestamps may be ambiguous due to daylight-savings time. For
                example in Central European Time (UTC+01), when going from
                03:00 DST to 02:00 non-DST, 02:30:00 local time occurs both at
                00:30:00 UTC and at 01:30:00 UTC. In such a situation, this
                parameter dictates how ambiguous times should be handled.

                - ``'infer'`` will attempt to infer fall dst-transition hours
                  based on order.
                - ``'NaT'`` will return NaT where there are ambiguous times.
                - ``'raise'`` will raise :class:`pandas.AmbiguousTimeError` if
                  there are ambiguous times.
                - An array of bools where True signifies a DST time, False
                  signifies a non-DST time (note that this flag is only
                  applicable for ambiguous times).
            name (Any):
                A name to be stored with the object.
        '''
        # Delegate to Pandas.
        index = pd.DatetimeIndex(data, tz=tz, ambiguous=ambiguous, name=name)

        # Ensure the timezone is UTC.
        if index.tz is None:
            tz = tz or 'UTC'
            index = index.tz_localize(tz).tz_convert('UTC')
        else:
            assert tz is None, 'Cannot pass tz when data is timezone aware'
            index = index.tz_convert('UTC')

        # Ensure the result is an instance of this class.
        index = super().__new__(cls, index)
        return index


def date_range(start=None, end=None, periods=None, freq=None, tz=None,
        normalize=False, name=None, closed=None):
    '''Return a fixed frequency DatetimeIndex.

    This is like :func:`pandas.date_range` but returns a timezone-aware
    :class:`apollo.DatetimeIndex`.

    When constructing an Apollo index, if the timezone would be ambiguous, it
    is assumed to be UTC. Otherwise it is converted to UTC.

    When creating a DatetimeIndex from strings, the special strings ``'now'``
    and ``'today'`` will almost certainly be misinterpreted. Always use true
    timestamps. The :class:`apollo.Timestamp` class will properly interpret
    the special strings.

    Arguments:
        start (str or datetime-like or None):
            Left bound for generating dates.
        end (str or datetime-like or None):
            Right bound for generating dates.
        periods (integer or None):
            Number of periods to generate.
        freq (str or DateOffset, default 'D'):
            The frequency of the range. Frequency strings can have multiples,
            e.g. ‘5H’. See Pandas for a list of frequency aliases.
        tz (str or pytz.timezone, default 'UTC'):
            The timezone in which to interpret ambiguous timestamps.
        normalize (bool):
            Normalize start/end dates to midnight before generating date range.
        name (Any):
            Name of the resulting :class:`DatetimeIndex`.
        closed ('left' or 'right' or None):
            Make the interval closed with respect to the given frequency to the
            'left', 'right', or both sides.
    '''
    # Interpret start and end with the same logic as `apollo.Timestamp`.
    # This handles the strings 'now' and 'today' and enables mixed timezones.
    if start is not None: start = Timestamp(start, tz=tz)
    if end is not None: end = Timestamp(end, tz=tz)

    # Delegate to Pandas, then ensure the index is an `apollo.DatetimeIndex`.
    # Do not pass `tz`; it has already been applied.
    index = pd.date_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        normalize=normalize,
        name=name,
        closed=closed,
    )
    return DatetimeIndex(index)


def time_of_day(times):
    '''Compute sine and cosine with a period of 24 hours.

    Arguments:
        times (numpy.ndarray like):
            A series of timestamps.

    Returns:
        pandas.DataFrame:
            A data frame with 2 data variables:

            - ``time_of_day_sin`` for the sine component of time-of-day.
            - ``time_of_day_cos`` for the cosine component of time-of-day.

            The data frame is indexed by the input timestamps.
    '''
    times = DatetimeIndex(times, name='time')  # Ensure UTC.
    nanos = np.asarray(times, dtype='datetime64[ns]').astype('float32')
    seconds = nanos / 1e9
    days = seconds / 86400

    return pd.DataFrame({
        'time_of_day_sin': np.sin(days * 2 * np.pi),
        'time_of_day_cos': np.cos(days * 2 * np.pi),
    }, index=times)


def time_of_year(times):
    '''Compute sine and cosine with a period of 365.25 days.

    The time-of-year is measured in Julian years, exactly 365.25 days of 86400
    seconds each. These are not exactly equal to a Gregorian year (a calendar
    year) once you get into the crazy leap rules.

    Arguments:
        times (numpy.ndarray like):
            A series of timestamps.

    Returns:
        pandas.DataFrame:
            A data frame with 2 data variables:

            - ``time_of_year_sin`` for the sine component of time-of-year.
            - ``time_of_year_cos`` for the cosine component of time-of-year.

            The data frame is indexed by the input timestamps.
    '''
    times = DatetimeIndex(times, name='time')  # Ensure UTC.
    nanos = np.asarray(times, dtype='datetime64[ns]').astype('float32')
    seconds = nanos / 1e9
    days = seconds / 86400
    years = days / 365.25

    return pd.DataFrame({
        'time_of_year_sin': np.sin(years * 2 * np.pi),
        'time_of_year_cos': np.cos(years * 2 * np.pi),
    }, index=times)


def is_daylight(times, lat, lon):
    '''Determine if the sun is above the horizon.

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
