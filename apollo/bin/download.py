#!/usr/bin/env python3
'''Download NAM forecasts.

This script is intended to be run as a cron job. Note that it imports
from `apollo.datasets`, so you have to make sure the package is in your
PYTHONPATH. The easiest way is to `cd` into this repository.

We run a cron job similar to the following to sync the local store:

    #!/bin/sh
    cd /mnt/data6tb/chris/
    date >> cron.log
    python3 -m bin.download -n 4 2>&1| tee -a ./download.log
'''

import argparse
import logging
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd

from apollo import storage, timestamps
from apollo.datasets import nam


def xrange_inclusive(start, stop, step):
    '''Like the builtin `range`, but:
        1. supports arbitrary data types, like datetime64, and
        2. is an inclusive range
    '''
    while start <= stop:
        yield start
        start += step


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Download NAM forecasts between two timestamps, inclusive.',
    )

    parser.add_argument(
        '-x',
        '--fail-fast',
        action='store_true',
        help='Do not retry downloads',
    )

    parser.add_argument(
        '-k',
        '--keep-gribs',
        action='store_true',
        help='do not delete grib files',
    )

    parser.add_argument(
        '-d',
        '--dest',
        type=str,
        help=f'path to the data store (default: {apollo.storage.get_root()})',
    )

    parser.add_argument(
        '-p',
        '--procs',
        type=int,
        default=1,
        help='use this many download processes (default: 1)',
    )

    parser.add_argument(
        '-n',
        '--count',
        type=int,
        help='download this many forecasts, ending at the reftime',
    )

    parser.add_argument(
        '-r',
        '--from',
        type=str,
        dest='start',  # Note that `--from` is parsed into `args.start`.
        help='download forecasts starting from this timestamp'
    )

    parser.add_argument(
        'reftime',
        nargs='?',
        default='now',
        help='the timestamp of the last forecast to download (default: now)',
    )

    args = parser.parse_args(argv)

    if args.count and args.start:
        print('The arguments --count/-n and --from/-r are mutually exclusive.')
        sys.exit(1)

    logging.debug('called with the following options:')
    for arg, val in vars(args).items():
        logging.debug(f'  {arg}: {val}')

    if args.dest:
        storage.set_root(args.dest)

    step = pd.Timedelta(6, 'h')
    stop = timestamps.utc_timestamp(args.reftime).floor('6h')
    if args.start:
        start = timestamps.utc_timestamp(args.start).floor('6h')
    elif args.count:
        start = stop - (args.count - 1) * step
    else:
        start = stop

    logging.info(f'downloading forecasts from {start} to {stop}')

    def download(reftime):
        try:
            nam.open(reftime, fail_fast=args.fail_fast, keep_gribs=args.keep_gribs)
        except Exception as e:
            logging.error(e)
            logging.error(f'Could not load data for {reftime}')

    # The argument `maxtasksperchild` causes the worker process to be destroyed
    # and restarted after fufilling this many tasks. This is defensive against
    # resource leaks in the NAM loader which may have existed with older xarray
    # versions.
    with mp.Pool(args.procs, maxtasksperchild=1) as pool:
        reftimes = xrange_inclusive(start, stop, step)
        pool.map(download, reftimes)


if __name__ == '__main__':
    main()
