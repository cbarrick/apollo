def description():
    import textwrap
    return textwrap.dedent('''\
    Execute an Apollo model.

    This command makes predictions for times between START and STOP at 1-hour
    intervals. Both START and STOP must be given as ISO 8601 timestamps and are
    rounded down to the hour. Timestamps are assumed to be UTC unless a
    timezone is given.

    If the --from-file/-f option is given, MODEL must be a path to a model file.
    Otherwise, MODEL must be the name of a model in the Apollo database.
    ''')


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description=description(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-f',
        '--from-file',
        action='store_true',
        help='load the model from a file',
    )

    parser.add_argument(
        'model',
        metavar='MODEL',
        help='the model to execute'
    )

    parser.add_argument(
        'start',
        metavar='START',
        help='the first timestamp'
    )

    parser.add_argument(
        'stop',
        nargs='?',
        metavar='STOP',
        help='the last timestamp (defaults to a 24-hour forecast)'
    )

    return parser.parse_args(argv)


def load_model(args):
    '''Loads the model from CLI arguments.
    '''
    import sys
    import apollo
    from apollo import models

    try:
        if args.from_file:
            return models.load_model_at(args.model)
        else:
            return models.load_model(args.model)
    except FileNotFoundError:
        print(
            f'No such model: {args.model}\n' +
            f'Hint: You can list known models with `apollo ls models`',
            file=sys.stderr,
        )
        sys.exit(1)


def get_times(args):
    '''Load the target timestamps from CLI arguments.
    '''
    import pandas as pd
    import apollo

    start = apollo.Timestamp(args.start).floor('1H')

    if args.stop is None:
        stop = start + pd.Timedelta(24, 'H')
    else:
        stop = apollo.Timestamp(args.stop).floor('1H')

    return apollo.date_range(start, stop, freq='1H')


def main(argv):
    import sys
    args = parse_args(argv)
    model = load_model(args)
    times = get_times(args)
    predictions = model.predict(times)
    predictions.to_csv(sys.stdout)
