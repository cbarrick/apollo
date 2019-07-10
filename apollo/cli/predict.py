def description():
    return 'Execute an Apollo model.'


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description=description()
    )

    parser.add_argument(
        '--named',
        '-n',
        action='store_true',
        help='Load a named model from the Apollo database.'
    )

    parser.add_argument(
        'model',
        metavar='MODEL',
        help='The model to execute.'
    )

    target_options = parser.add_mutually_exclusive_group()

    target_options.add_argument(
        '--range',
        '-r',
        nargs=2,
        metavar=['START', 'STOP'],
        help='Make predictions for a range of times.'
    )

    target_options.add_argument(
        '--csv',
        '-c',
        metavar='FILE',
        help='Make predictions for times listed as the first column in a CSV.'
    )

    return parser.parse_args(argv)


def load_model(args):
    '''Loads the model from CLI arguments.

    If the ``-n/--named`` argument is given, we load a named model from the
    Apollo database.

    Otherwise we load a model from a JSON template file.
    '''
    import sys
    from pathlib import Path

    import apollo
    from apollo import models
    from apollo.models import Model

    if args.named:
        name = args.model
        known_models = models.list_models()
        if name not in known_models:
            print(f'Unknown model: {name}', file=sys.stderr)
            print(f'Hint: use `apollo ls models` to list models', file=sys.stderr)
            sys.exit(1)
        return Model.load_named(args.model)

    else:
        path = Path(args.model)
        if not path.exists():
            print(f'No such file: {path}', file=sys.stderr)
            sys.exit(1)
        return Model.load(args.model)


def load_times(args):
    '''Load the target timestamps from CLI arguments.

    If the ``-r/--range`` argument is given, we construct a range of timestamps
    at 1-hour frequency.

    Otherwise we read a CSV and parse the first column as for timestamps.
    '''
    import apollo
    import pandas as pd
    import sys

    if args.range:
        (start, stop) = args.range
        return apollo.date_range(start, stop, freq='1H')

    else:
        if args.csv == '-': args.csv = sys.stdin
        times = pd.read_csv(
            args.csv,
            parse_dates=True,
            usecols=[0],
            index_col=0,
        )
        return times.index


def main(argv):
    import sys
    args = parse_args(argv)
    model = load_model(args)
    times = load_times(args)
    prediction = model.predict(times)
    predictions.to_csv(sys.stdout)
