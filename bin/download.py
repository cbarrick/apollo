#!/usr/bin/env python3
'''Download missing NAM forecasts to the cache.

This script is intended to be run as a cron job. Note that it imports
from `ugasolar.datasets`, so you have to make sure the package is in your
PYTHONPATH. The easiest way is to `cd` into this repository.

We run a cron job similar to the following to keep the cache updated:

    #!/bin/sh
    cd /mnt/data6tb/chris/
    date >> cron.log
    python3 -m bin.download -n 4 2>&1| tee -a ./download.log
'''
from ugasolar.datasets import nam

import argparse
import logging
import sys

import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download missing NAM forecasts to the cache.')
    parser.add_argument('-c', '--cache-dir', type=str, default='./data/NAM-NMM', help='Path to the cache.')
    parser.add_argument('-n', '--count', type=int, default=1, metavar='N', help='Download N datasets, ending at the reference time.')
    parser.add_argument('-x', '--fail-fast', action='store_true', help='Do not retry downloads.')
    parser.add_argument('-k', '--keep-gribs', action='store_true', help='Do not delete grib files.')
    parser.add_argument('time', nargs='?', default='now', help='The reference time to download.')
    args = parser.parse_args()

    logging.basicConfig(format='[{asctime}] {levelname}: {message}', style='{')
    nam.logger.setLevel('DEBUG')

    if args.count < 1:
        logging.error(f'Count must be greater than 0, got {args.count}')
        sys.exit(1)

    reftime = np.datetime64(args.time, '6h')

    for i in range(args.count):
        try:
            nam.open(
                reftime,
                cache_dir=args.cache_dir,
                fail_fast=args.fail_fast,
                keep_gribs=args.keep_gribs)
        except Exception as e:
            logging.error(e)
            logging.error(f'Could not load data from {reftime}')

        reftime -= np.timedelta64(6, 'h')
