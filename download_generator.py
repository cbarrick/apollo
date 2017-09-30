#!/usr/bin/env python3
'''Generates a script to download missing forecasts to the cache.'''

from datetime import datetime, timedelta, timezone
from pathlib import Path


def xrange(start, stop, step):
    while start < stop:
        yield start
        start += step


def reftime(t):
    # Convert strings
    if isinstance(t, str):
        t = datetime.strptime(t, '%Y%m%dT%H%M')
        t = t.replace(tzinfo=timezone.utc)

    # Convert to UTC
    t = t.astimezone(timezone.utc)

    # Round to the previous 0h, 6h, 12h, or 18h
    hour = (t.hour // 6) * 6
    t = t.replace(hour=hour, minute=0, second=0, microsecond=0)

    return t


def missing_datasets(start, stop, basepath='.'):
    start = reftime(start)
    stop = reftime(stop)
    step = timedelta(hours=6)
    basepath = Path(basepath)
    for i, t in enumerate(xrange(start, stop, step)):
        filename = f'nam.{t.year}{t.month:02}{t.day:02}/nam.t{t.hour:02}z.awphys.tm00.nc'
        path = basepath / filename
        if not path.exists():
            yield t


def main(start=None, stop=None, log='missing.log', basepath=None):
    start = start or '20160901T0000'
    stop = stop or datetime.now(tz=timezone.utc)
    basepath = basepath or '.'
    for i, t in enumerate(missing_datasets(start, stop, basepath)):
        spec = f'{t.year}{t.month:02}{t.day:02}T{t.hour:02}00'
        print('./download.py', '-t', spec, '-f', 36, '-x', '2>&1 | tee -a', log)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate scripts to download missing datasets.')
    parser.add_argument('start', type=reftime, help='Start of the range (%Y%m%dT%H%M).')
    parser.add_argument('stop',  type=reftime, help='End of the range (%Y%m%dT%H%M).')
    parser.add_argument('--log', type=str, default='missing.log', help='Path of the log file.')
    args = parser.parse_args()
    main(**vars(args))
