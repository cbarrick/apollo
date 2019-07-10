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


def description():
    import textwrap
    return textwrap.dedent('''\
    Download and process NAM forecasts.

    Forecasts are selected by one of --reftime/-t, --range/-r, or --count/-n.
    If none of those options are provided, the most recent forecast is selected.
    ''')


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description=description(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-x',
        '--fail-fast',
        action='store_true',
        help='do not retry downloads',
    )

    parser.add_argument(
        '-k',
        '--keep-gribs',
        action='store_true',
        help='do not delete grib files',
    )

    selectors = parser.add_mutually_exclusive_group()

    selectors.add_argument(
        '-t',
        '--reftime',
        metavar='TIMESTAMP',
        help='download the forecast for the given reftime',
    )

    selectors.add_argument(
        '-r',
        '--range',
        nargs=2,
        metavar=('START', 'STOP'),
        help='download all forecast on this range, inclusive'
    )

    selectors.add_argument(
        '-n',
        '--count',
        type=int,
        metavar='N',
        help='download the N most recent forecasts',
    )

    return parser.parse_args(argv)


def reftimes(args):
    '''Iterate over the reftimes specified by the command-line arguments.

    Yields:
        Timestamp:
            A timestamp for the reftime.
    '''
    import pandas as pd

    import apollo

    import logging
    logger = logging.getLogger(__name__)

    # The ``reftime`` mode gives a single reftime.
    if args.reftime is not None:
        reftime = apollo.Timestamp(args.reftime)
        logger.info(f'selected the forecast for reftime {reftime}')
        yield reftime

    # The ``range`` mode gives the reftime between two inclusive endpoints.
    elif args.range is not None:
        start = apollo.Timestamp(args.range[0])
        stop = apollo.Timestamp(args.range[1])
        step = pd.Timedelta(6, 'h')
        logger.info(f'selected the forecasts between {start} and {stop} (inclusive)')
        while start <= stop:
            yield start
            start += step

    # The ``count`` mode downloads the N most recent reftimes.
    elif args.count is not None:
        n = args.count
        reftime = apollo.Timestamp('now').floor('6h')
        step = pd.Timedelta(6, 'h')
        logger.info(f'selected the {n} most recent forecasts (ending at {reftime})')
        for _ in range(n):
            yield reftime
            reftime -= step

    # The default is to use the most recent reftime.
    else:
        reftime = apollo.Timestamp('now').floor('6h')
        logger.info(f'selected the most recent forecast ({reftime})')
        yield reftime


def main(argv=None):
    import apollo
    from apollo.datasets import nam

    import logging
    logger = logging.getLogger(__name__)

    args = parse_args(argv)

    logger.debug('called with the following options:')
    for arg, val in vars(args).items():
        logger.debug(f'  {arg}: {val}')

    for reftime in reftimes(args):
        try:
            nam.download(reftime, fail_fast=args.fail_fast, keep_gribs=args.keep_gribs)
        except Exception as e:
            logger.error(e)
            logger.error(f'Could not download data for {reftime}')
