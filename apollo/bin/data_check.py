import argparse
import logging
import sys
from collections import namedtuple
from pathlib import Path

import pandas as pd

import apollo.storage
from apollo.datasets import nam


def reftimes(args):
    '''Iterate over the reftime range selected by the command line.
    '''
    if args.count and args.start:
        print('The arguments --count/-n and --from/-r are mutually exclusive.')
        sys.exit(1)

    step = pd.Timedelta(6, 'h')

    stop = pd.Timestamp(args.reftime).floor('6h')

    if args.start:
        start = pd.Timestamp(args.start).floor('6h')
    elif args.count:
        start = stop - (args.count - 1) * step
    else:
        start = stop

    size = (stop - start) // step

    logging.debug(f'selected reftimes from {start} to {stop} ({size} forecasts)')

    while start <= stop:
        yield start
        start += step

def local_reftimes(args):
    '''Iterate over the reftimes for which we have data.
    '''
    for reftime in reftimes(args):
        try:
            nam.open_local(reftime)
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
            data_a, data_b = nam.open_local(a), nam.open_local(b)
            yield data_a, data_b
            data_a.close(); data_b.close()
            data_a, data_b = nam.open_local(a), nam.open_local(b)
            yield data_b, data_a
            a = b
        except StopIteration:
            return


def nc_path(reftime):
    '''The path for a local netCDF file.
    '''
    loader = nam.NamLoader()
    return loader.nc_path(reftime)


if __name__ == '__main__':
    # Note that the `--from` argument is parsed into `args.start`
    parser = argparse.ArgumentParser(description='Check NAM forecasts for join bugs.')
    parser.add_argument('-d', '--dry-run', action='store_true', help='Do not prompt to fix errors.')
    parser.add_argument('-s', '--store', type=str, help='Path to the data store.')
    parser.add_argument('-n', '--count', type=int, help='Check this many forecasts, defaults to 2.')
    parser.add_argument('-r', '--from', type=str, dest='start', help='Check forecasts starting from this reftime.')
    parser.add_argument('-l', '--log', type=str, default='INFO', help='Set the log level.')
    parser.add_argument('reftime', nargs='?', default='now', help='The timestamp to check. Defaults to the most recent.')
    args = parser.parse_args()

    logging.basicConfig(
        format='[{asctime}] {levelname}: {message}',
        style='{',
        level=args.log.upper(),
    )
    nam.logger.setLevel(args.log.upper())

    logging.debug('called with the following options:')
    for arg, val in vars(args).items():
        logging.debug(f'  {arg}: {val}')

    if args.store:
        apollo.storage.set_root(args.store)

    now = pd.Timestamp('now', tz='utc')

    for (a, b) in dataset_pairs(args):
        time_a = pd.Timestamp(a.reftime.data[0]).floor('6h')
        time_b = pd.Timestamp(b.reftime.data[0]).floor('6h')
        vars_a = set(a.variables.keys())
        vars_b = set(b.variables.keys())
        path_a = nc_path(time_a)
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
