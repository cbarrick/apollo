#!/usr/bin/env python3
from uga_solar.nam import data

import argparse
import logging
import sys
from datetime import datetime, timezone


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
    parser = argparse.ArgumentParser(description='Download NAM forecasts to the cache.')
    parser.add_argument('-c', '--cache-dir', type=str, default='./NAM-NMM', help='Path to the cache.')
    parser.add_argument('-n', '--count', type=int, default=1, metavar='N', help='Download N datasets, ending at the reference time.')
    parser.add_argument('-x', '--fail-fast', action='store_true', help='Do not retry downloads.')
    parser.add_argument('-k', '--keep-gribs', action='store_true', help='Do not delete grib files.')
    parser.add_argument('-l', '--log', type=str, default='INFO', help='Set the log level.')
    parser.add_argument('time', type=reftime, help='The reference time to download.')
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log,
        style='{',
        format='[{asctime}] {levelname}: {message}')

    if args.count < 1:
        logging.error('Count must be greater than 0, got {}'.format(args.count))
        sys.exit(1)

    reftime = args.time or datetime.now(timezone.utc)

    for i in range(args.count):
        try:
            data.load_nam(
                reftime,
                cache_dir=args.cache_dir,
                fail_fast=args.fail_fast,
                save_gribs=args.keep_gribs)
        except Exception as e:
            logger.error(e)
            logger.error('Could not load data from {}'.format(reftime))

        reftime -= timedelta(hours=6)
