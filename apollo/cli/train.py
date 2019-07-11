def description():
    return 'Train a new model'


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description=description()
    )

    parser.add_argument(
        '--name',
        metavar='NAME',
        default=None,
        help='a name for the model, defaults to a UUID',
    )

    parser.add_argument(
        '-t',
        dest='named_template',
        action='store_true',
        help='load a builtin template rather than a template file',
    )

    parser.add_argument(
        '-o',
        metavar='DEST',
        dest='dest',
        default=None,
        help='output path for the model, defaults to the Apollo database',
    )

    parser.add_argument(
        'template',
        metavar='TEMPLATE',
        help='a path to template file or the name of a builtin template',
    )

    parser.add_argument(
        'data_file',
        metavar='DATA',
        nargs='?',
        default=sys.stdin,
        help='the training data (a CSV, defaults to stdin)'
    )

    return parser.parse_args(argv)


def main(argv):
    import sys

    import pandas as pd

    import apollo
    from apollo import models
    from apollo.data import ga_power

    import logging
    logger = logging.getLogger(__name__)

    args = parse_args(argv)

    logger.info('instantiating model from template')
    if args.named_template:
        model = models.from_named_template(args.template, name=args.name)
    else:
        model = models.from_template(args.template, name=args.name)

    logger.info('reading training data')
    targets = pd.read_csv(args.data_file, parse_dates=True, index_col=0)

    logger.info(f'training model with name {model.name}')
    model.fit(targets)
    path = model.save(args.dest)
    print(path)
