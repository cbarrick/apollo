def description():
    import textwrap
    return textwrap.dedent('''\
    Syncronize the geo coordinates for all NAM forecasts.

    If NOAA changes the way geographic coordinates are computed, the coordinate
    in the new forecasts will not align with the coordinates in the old
    forecasts. This happened in March of 2017 when NOAA migrated from the
    GRIB1 format to the GRIB2 format.

    This command identifies forecasts whose coordinates do not match the most
    recent forecast and overwrites the old coordinates with the new values.
    This transformation is only applied if the old coordinates are within 12 km
    of the new values. A greater difference implies that the grids of the
    forecasts do not correspond to the same position.
    ''')


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description=description(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-d',
        '--dry-run',
        action='store_true',
        help='only print the actions that would be taken'
    )

    selectors = parser.add_mutually_exclusive_group()

    selectors.add_argument(
        '-r',
        '--range',
        nargs=2,
        metavar=('START', 'STOP'),
        help='only consider forecasts within this range of reftimes, inclusive'
    )

    selectors.add_argument(
        '-n',
        '--count',
        type=int,
        metavar='N',
        help='consider only the N most recent forecasts',
    )

    return parser.parse_args(argv)


def iter_reftimes(args):
    '''Iterate over the reftimes specified by the command-line arguments.

    Yields:
        Timestamp:
            A timestamp for the reftime.
    '''
    import pandas as pd

    import apollo

    import logging
    logger = logging.getLogger(__name__)

    # The ``range`` mode yields the reftimes between two inclusive endpoints.
    if args.range is not None:
        start = apollo.Timestamp(args.range[0])
        stop = apollo.Timestamp(args.range[1])
        step = pd.Timedelta(6, 'h')
        logger.info(f'selected the forecasts between {start} and {stop} (inclusive)')
        while start <= stop:
            yield start
            start += step

    # The ``count`` mode yields the N most recent reftimes.
    elif args.count is not None:
        n = args.count
        reftime = apollo.Timestamp('now').floor('6h')
        step = pd.Timedelta(6, 'h')
        logger.info(f'selected the {n} most recent forecasts (ending at {reftime})')
        for _ in range(n):
            yield reftime
            reftime -= step

    # The default yields all available reftimes.
    else:
        logger.info(f'selected all forecasts')
        yield from nam.iter_available_forecasts()


def main(argv):
    import sys
    from apollo.data import nam

    import logging
    logger = logging.getLogger(__name__)

    args = parse_args(argv)

    reftimes = sorted(iter_reftimes(args))
    latest = nam.open(reftimes[-1]).load()

    for reftime in reftimes:
        logger.info(f'Checking forecast: {reftime}')
        try:
            data = nam.open(reftime, on_miss='raise')
        except nam.CacheMiss:
            logger.warning(f'Missing forecast: {reftime}')
        ok_x = (data.x.values == latest.x.values).all()
        ok_y = (data.y.values == latest.y.values).all()
        if ok_x and ok_y: continue

        print(f'Needs alignment: {reftime}')

        syncable_x = (abs(data.x.values - latest.x.values) < 12000).all()
        syncable_y = (abs(data.y.values - latest.y.values) < 12000).all()
        if not syncable_x or not syncable_y:
            # This means the geo coordinates are off by more than 12 km.
            # We cannot realign in this case. This case should be unreachable.
            print(f'ERROR: Cannot align')
            print(abs(data.x.values - latest.x.values))
            print(abs(data.y.values - latest.y.values))
            sys.exit(1)

        if not args.dry_run:
            print('Aligning...', end='', flush=True)
            path = nam.nc_path(reftime)
            data['x'] = latest.x
            data['y'] = latest.y
            data.load().close()
            data.to_netcdf(path)
            print('done')
