#!/usr/bin/env python3
'''Fixes a metadata error.

The main dataloader gets its forecast times from metadata in the grib files
or as part of the file name. A previous bug would mistakenly set the forecast
time (the time bein predicted) to the validity time (the time the forecast was
made). This was only a problem for older grib files that did not include the
'forecastTime' key.

This script sets the identifies this bug and resets the forecast coordinate to
the official forecast cycle.
'''

from datetime import datetime, timedelta, timezone
from pathlib import Path
import itertools

import numpy as np
import xarray as xr


def xrange(start, stop, step):
    while start < stop:
        yield start
        start += step


def reftime(reftime):
    # Convert strings
    if isinstance(reftime, str):
        reftime = datetime.strptime(reftime, '%Y%m%dT%H%M')
        reftime = reftime.replace(tzinfo=timezone.utc)

    # Convert to UTC
    reftime = reftime.astimezone(timezone.utc)

    # Round to the previous 0h, 6h, 12h, or 18h
    hour = (reftime.hour // 6) * 6
    reftime = reftime.replace(hour=hour, minute=0, second=0, microsecond=0)

    return reftime


def datasets(start, stop, basepath='.'):
    start = reftime(start)
    stop = reftime(stop)
    step = timedelta(hours=6)
    basepath = Path(basepath)
    for i, t in enumerate(xrange(start, stop, step)):
        filename = f'nam.{t.year}{t.month:02}{t.day:02}/nam.t{t.hour:02}z.awphys.tm00.nc'
        path = basepath / filename
        if path.exists():
            yield path


def is_broken(ds):
    forecast = ds['forecast'].values
    return forecast.astype(int).std() == 0


def proper_forecast(n):
    start = np.timedelta64(0, 'h')
    pivot = np.timedelta64(36, 'h')
    stop = np.timedelta64(85, 'h')
    step_small = np.timedelta64(1, 'h')
    step_big = np.timedelta64(3, 'h')
    high_freq = xrange(start, pivot, step_small)
    low_freq = xrange(pivot, stop, step_big)
    forecast = itertools.chain(high_freq, low_freq)
    forecast = itertools.islice(forecast, n)
    forecast = xr.Variable('forecast', list(forecast))
    return forecast


def reset_forecast(ds):
    forecast = ds['forecast'].values
    n = len(forecast)
    ds['forecast'] = proper_forecast(n)
    return ds


def move_to_backup(path):
    backup = Path(str(path) + '.bak')
    path.rename(backup)
    return backup


def main(start='20160901T0000', stop='20170420T0000', basepath='.'):
    for path in datasets(start, stop, basepath):
        print(f'Checking {path}')
        ds = xr.open_dataset(str(path))
        if is_broken(ds):
            print(f'Fixing {path}', end='...', flush=True)
            ds.load()
            backup = move_to_backup(path)
            reset_forecast(ds)
            ds.to_netcdf(str(path))
            backup.unlink()
            print('DONE')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fix forecast axis in broken datasets.')
    parser.add_argument('start', type=reftime, nargs='?', help='Start of the range (%Y%m%dT%H%M).')
    parser.add_argument('stop',  type=reftime, nargs='?', help='End of the range (%Y%m%dT%H%M).')
    args = parser.parse_args()
    main(**vars(args))
