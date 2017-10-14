#!/usr/bin/env python3
'''Download missing NAM forecasts to the cache.

This script is intended to be run as a cron job. Note that it imports
from `uga_solar`, so you have to make sure the package is in your
PYTHONPATH. The easiest way is to `cd` into this repository.

We run the following script as a cron job to keep the cache updated:

    #!/bin/sh
    cd /mnt/data6tb/chris/
    date >> cron.log
    ./download.py -n 4 2>&1| tee -a ./download.log
'''
from uga_solar.data import nam

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download missing NAM forecasts to the cache.')
    parser.add_argument('-c', '--cache-dir', type=str, default='./NAM-NMM', help='Path to the cache.')
    parser.add_argument('-n', '--count', type=int, default=1, metavar='N', help='Download N datasets, ending at the reference time.')
    parser.add_argument('-x', '--fail-fast', action='store_true', help='Do not retry downloads.')
    parser.add_argument('-k', '--keep-gribs', action='store_true', help='Do not delete grib files.')
    parser.add_argument('time', nargs='?', type=reftime, help='The reference time to download.')
    args = parser.parse_args()

    logging.basicConfig(format='[{asctime}] {levelname}: {message}', style='{')
    nam.logger.setLevel('DEBUG')

    if args.count < 1:
        logging.error('Count must be greater than 0, got {}'.format(args.count))
        sys.exit(1)

    reftime = args.time or datetime.now(timezone.utc)

    for i in range(args.count):
        try:
            nam.load(
                reftime,
                cache_dir=args.cache_dir,
                fail_fast=args.fail_fast,
                keep_gribs=args.keep_gribs)
        except Exception as e:
            logging.error(e)
            logging.error('Could not load data from {}'.format(reftime))

        reftime -= timedelta(hours=6)
