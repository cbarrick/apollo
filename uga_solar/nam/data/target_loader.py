from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import sys

import pandas as pd

logger = logging.getLogger(__name__)


def load_targets(module, basepath='GA-POWER', **kwargs):
    kwargs.setdefault('header', None)
    kwargs.setdefault('index_col', [0])
    kwargs.setdefault('parse_dates', [0])
    kwargs.setdefault('infer_datetime_format', True)

    def parts():
        for path in paths:
            try:
                logger.debug(f'Reading {path}')
                df = pd.read_csv(str(path), **kwargs)
                df = df.tz_localize('UTC')
                yield df
                continue
            except Exception as e:
                logger.warning(f'Could not read {path}')
                logger.warning(e)

            try:
                logger.warning('Retrying without compression')
                df = pd.read_csv(str(path), **kwargs, compression=None)
                df = df.tz_localize('UTC')
                yield df
                continue
            except Exception as e:
                logger.error(f'Could not read {path}')
                logger.error(e)
                logger.error('skipping file')

    logger.info(f'Loading module {int(module)}')
    basepath = Path(basepath)
    paths = basepath.glob(f'**/mb-{int(module):03}.*.log.gz')
    return pd.concat(parts())


def pool(df, **kwargs):
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

    logger.info('Downsampling')
    return df.groupby(time).mean()


def save(df,
         module,
         basepath='GA-POWER',
         fmt='mb-{module:03}-targets.csv',
         **kwargs):
    kwargs.setdefault('header', False)
    dest = basepath / fmt.format(module=module)
    logger.info(f'Writing {dest}')
    df.to_csv(str(dest), **kwargs)


def main(modules=[1, 2, 3, 4, 5, 6, 7, 8], basepath='GA-POWER'):
    basepath = Path(basepath)

    try:
        len(modules)
    except TypeError:
        modules = [modules]

    for mod in modules:
        df = load_targets(mod, basepath)
        df = pool(df, minute=15)
        save(df, mod, basepath)
