#!/usr/bin/env python3
'''Generates a script to download missing forecasts to the cache.

This script enumerates all missing NAM forecasts between two times.
It prints a script that calls into `download.py` separately for each
missing forecast. This is more flexible than using `download.py`
directly when managing many downloads or a spotty cache.
'''
from pathlib import Path

import numpy as np


def xrange(start, stop, step):
    while start < stop:
        yield start
        start += step


def reftime(t):
    return np.datetime64(t, '6h')


def missing_datasets(start, stop, basepath='.'):
    start = reftime(start)
    stop = reftime(stop)
    step = np.timedelta64(6, 'h')
    basepath = Path(basepath)
    for i, t in enumerate(xrange(start, stop, step)):
        filename = 'nam.{t.year}{t.month:02}{t.day:02}/nam.t{t.hour:02}z.awphys.tm00.nc'
        filename = filename.format(t=t.astype(object))
        path = basepath / filename
        if not path.exists():
            yield t


def main(start='2017-01-01', stop='today', log='missing.log', basepath=None):
    start = reftime(start)
    stop = reftime(stop)
    basepath = basepath or '.'
    for t in missing_datasets(start, stop, basepath):
        print('./download.py', '-x', t, '2>&1 | tee -a', log)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate scripts to download missing datasets.')
    parser.add_argument('start', default='2017-01-01', help='Start of the range (%Y%m%dT%H%M).')
    parser.add_argument('stop',  default='today', help='End of the range (%Y%m%dT%H%M).')
    parser.add_argument('--log', type=str, default='missing.log', help='Path of the log file.')
    args = parser.parse_args()
    main(**vars(args))
