from datetime import datetime
from pathlib import Path
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def pool(df, **kwargs):
    '''Aggregates a DataFrame by time.

    This function takes in a mapping from a time unit to a multiple. Each row
    of the dataframe is grouped by rounding that time unit to the previous
    multiple, e.g. `minute=15` groups the data into 15 minute blocks.

    The average is then taken for each group.

    Args:
        year (int): Group by rounding the year.
        month (int): Group by rounding the month.
        day (int): Group by rounding the day.
        hour (int): Group by rounding the hour.
        minute (int): Group by rounding the minute.

    Returns:
        The average values for each group.
    '''
    kwargs.setdefault('year', 1)
    kwargs.setdefault('month', 1)
    kwargs.setdefault('day', 1)
    kwargs.setdefault('hour', 1)
    kwargs.setdefault('minute', 1)

    def round_down(num, divisor):
        return num - (num % divisor)

    def time(t):
        year = round_down(t.year, kwargs['year'])
        month = round_down(t.month, kwargs['month'])
        day = round_down(t.day, kwargs['day'])
        hour = round_down(t.hour, kwargs['hour'])
        minute = round_down(t.minute, kwargs['minute'])
        return datetime(year, month, day, hour, minute, tzinfo=t.tzinfo)

    return df.groupby(time).mean()


class TargetLoader:
    '''A data loader for the GA Power target data.

    The data should live together in some directory with names matching the
    pattern: `**/mb-{module:03}.*.log.gz' where `module` is an integer index.

    The loader will compute agregate statistics for 15 minute windows and
    cache the results in the same directory.
    '''

    def __init__(self,
            data_dir='./GA-POWER',
            data_fmt='mb-{module:03}-targets.csv'):
        '''Create a new TargetLoader.

        Args:
            data_dir (Path or str):
                The path to the directory containing the data.
            data_fmt (str):
                The filename format for saving/loading summary statistics.
        '''
        self.data_dir = Path(data_dir)
        self.data_fmt = data_fmt

    def load_raw_targets(self, module):
        read_options = {
            'header': None,
            'index_col': [0],
            'parse_dates': [0],
            'infer_datetime_format': True,
        }

        def parts():
            for path in self.data_dir.glob(f'**/mb-{int(module):03}.*.log.gz'):
                try:
                    logger.debug(f'Reading {path}')
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

        logger.info(f'Loading module {int(module)}')
        return pd.concat(parts())

    def rebuild(self, module, **kwargs):
        write_options = {
            'header': False,
        }

        df = self.load_raw_targets(module)
        df = df.sort_index()
        df = pool(df, minute=15)
        dest = self.data_dir / self.data_fmt.format(module=module)
        logger.info(f'Writing {dest}')
        df.to_csv(str(dest), **write_options)

    def load_targets(self, module):
        read_options = {
            'header': None,
            'index_col': [0],
            'parse_dates': [0],
            'infer_datetime_format': True,
        }

        path = self.data_dir / self.data_fmt.format(module=module)
        if not path.exists():
            self.rebuild(module)

        return pd.read_csv(str(path), **read_options)


def load_targets(module, **kwargs):
    loader = TargetLoader(**kwargs)
    return loader.load_targets(module)
