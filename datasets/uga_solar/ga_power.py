from datetime import datetime
from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def open_aggregate(module, **kwargs):
    loader = GaPowerLoader(**kwargs)
    return loader.open_aggregate(module)


def round_down(num, divisor):
    return num - (num % divisor)


def interval(**kwargs):
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
    kwargs.setdefault('year', 1)
    kwargs.setdefault('month', 1)
    kwargs.setdefault('day', 1)
    kwargs.setdefault('hour', 1)
    kwargs.setdefault('minute', 1)

    def grouper(t):
        year = round_down(t.year, kwargs['year'])
        month = round_down(t.month, kwargs['month'])
        day = round_down(t.day, kwargs['day'])
        hour = round_down(t.hour, kwargs['hour'])
        minute = round_down(t.minute, kwargs['minute'])
        return datetime(year, month, day, hour, minute, tzinfo=t.tzinfo)

    return grouper


class GaPowerLoader:
    '''A database of the GA Power target data.

    The data should live together in some directory with names matching the
    pattern: `**/mb-{module:03}.*.log.gz' where `module` is an integer index.

    The database will compute agregate statistics for 1 hour windows and
    cache the results in the same directory.
    '''

    def __init__(self,
            data_dir='./data/GA-POWER',
            data_fmt='mb-{module:03}-targets.csv'):
        '''Create a new GaPowerLoader.

        Args:
            data_dir (Path or str):
                The path to the directory containing the data.
            data_fmt (str):
                The filename format for saving/loading summary statistics.
        '''
        self.data_dir = Path(data_dir)
        self.data_fmt = data_fmt

    def select(self, module):
        read_options = {
            'header': None,
            'index_col': [0],
            'parse_dates': [0],
            'infer_datetime_format': True,
        }

        def parts():
            for path in self.data_dir.glob(f'**/mb-{int(module):03}.*.log.gz'):
                try:
                    logger.info(f'Reading {path}')
                    df = pd.read_csv(str(path), **read_options)
                    df = df.tz_localize('UTC')
                    yield df
                    continue
                except Exception as e:
                    logger.warning(f'Could not read {path}')
                    logger.warning(e)

                try:
                    logger.warning('Retrying without compression')
                    df = pd.read_csv(str(path), **read_options, compression=None)
                    df = df.tz_localize('UTC')
                    yield df
                    continue
                except Exception as e:
                    logger.error(f'Could not read {path}')
                    logger.error(e)
                    logger.error('skipping file')

        logger.info(f'loading raw module {module}')
        return pd.concat(parts())

    def open_aggregate(self, module):
        read_options = {
            'header': None,
            'index_col': [0],
            'parse_dates': [0],
            'infer_datetime_format': True,
        }

        write_options = {
            'header': False,
        }

        path = self.data_dir / self.data_fmt.format(module=module)
        if not path.exists():
            df = self.select(module)
            logger.info(f'computing aggregate')
            df = df.sort_index()
            df = df.groupby(interval(minute=60)).mean()
            logger.info(f'writing to cache {path}')
            df.to_csv(str(path), **write_options)
        else:
            logger.info(f'loading aggregate module {module} from cache')
            df = pd.read_csv(str(path), **read_options)

        return df
