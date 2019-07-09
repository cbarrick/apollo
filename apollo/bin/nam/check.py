import argparse
import logging
import sys
from collections import namedtuple
from pathlib import Path

import pandas as pd

import apollo
from apollo import storage
from apollo.datasets import nam


def reftimes(args):
    '''Iterate over the reftimes specified by the command-line arguments.

    Yields:
        Timestamp:
            A timestamp for the reftime.
    '''
    # The ``reftime`` mode gives a single reftime.
    if args.reftime is not None:
        reftime = apollo.Timestamp(args.reftime)
        logging.info(f'selected the forecast for reftime {reftime}')
        yield reftime

    # The ``range`` mode gives the reftime between two inclusive endpoints.
    elif args.range is not None:
        start = apollo.Timestamp(args.range[0])
        stop = apollo.Timestamp(args.range[1])
        step = pd.Timedelta(6, 'h', tz='utc')
        logging.info(f'selected the forecasts between {start} and {stop} (inclusive)')
        while start <= stop:
            yield start
            start += step

    # The ``count`` mode downloads the N most recent reftimes.
    elif args.count is not None:
        n = args.count
        reftime = apollo.Timestamp('now').floor('6h')
        step = pd.Timedelta(6, 'h', tz='utc')
        logging.info(f'selected the {n} most recent forecasts (ending at {reftime})')
        for _ in range(n):
            yield reftime
            reftime -= step

    # The default is to use the most recent reftime.
    else:
        reftime = apollo.Timestamp('now').floor('6h')
        logging.info(f'selected the most recent forecast ({reftime})')
        yield reftime


def local_reftimes(args):
    '''Iterate over the reftimes for which we have data.
    '''
    for reftime in reftimes(args):
        try:
            nam.open(reftime)
        except nam.CacheMiss:
            continue
        yield reftime


def dataset_pairs(args):
    '''Iterate over adjacent pairs of datasets, both ways.
    '''
    # Reload the data every time since a fix may change the data on disk.
    it = iter(local_reftimes(args))
    a = next(it)
    while True:
        try:
            b = next(it)
            data_a, data_b = nam.open(a), nam.open(b)
            yield data_a, data_b
            data_a.close(); data_b.close()
            data_a, data_b = nam.open(a), nam.open(b)
            yield data_b, data_a
            a = b
        except StopIteration:
            return


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Check NAM forecasts for bugs.'
    )

    parser.add_argument(
        '-d',
        '--dry-run',
        action='store_true',
        help='identify errors but do not prompt to correct them'
    )

    selectors = parser.add_mutually_exclusive_group()

    selectors.add_argument(
        '-t',
        '--reftime',
        metavar='TIMESTAMP',
        help='check the forecast for the given reftime',
    )

    selectors.add_argument(
        '-r',
        '--range',
        nargs=2,
        metavar=('START', 'STOP'),
        help='check all forecast on this range, inclusive'
    )

    selectors.add_argument(
        '-n',
        '--count',
        type=int,
        metavar='N',
        help='check the N most recent forecasts',
    )

    args = parser.parse_args(argv)

    logging.debug('called with the following options:')
    for arg, val in vars(args).items():
        logging.debug(f'  {arg}: {val}')

    now = apollo.Timestamp('now', tz='utc')

    for (a, b) in dataset_pairs(args):
        time_a = apollo.Timestamp(a.reftime.data[0]).floor('6h')
        time_b = apollo.Timestamp(b.reftime.data[0]).floor('6h')
        vars_a = set(a.variables.keys())
        vars_b = set(b.variables.keys())
        path_a = nam.nc_path(time_a)
        path_backup = Path(f'{path_a}.bak')

        if vars_a - vars_b:
            diff = list(vars_a - vars_b)
            print(f'Variables found for {time_a} but not for {time_b}: {diff}')
            if not args.dry_run:
                fix = input(f'Delete these variables from {time_a} [y/N]? ')
                if fix.upper().startswith('Y'):
                    logging.info(f'backing up dataset to {path_backup}')
                    path_a.rename(path_backup)
                    logging.info(f'deleting spurious variables from {path_a}')
                    history = a.attrs['history']
                    if not history.endswith('\n'): history += '\n'
                    for var in diff:
                        history += f'{now.isoformat()} Delete variable {var}\n'
                    ds = a.load()
                    ds = ds.assign_attrs(history=history)
                    ds = ds.drop(diff)
                    ds.to_netcdf(path_a)
                    assert path_a.exists()


if __name__ == '__main__':
    main()
