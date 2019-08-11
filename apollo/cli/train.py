def description():
    import textwrap
    return textwrap.dedent('''\
    Train a new model.

    The training data is given as a CSV. The file must have a header row, and
    the first column must be the timestamps of the observations.

    Once trained, this command prints the path to the serialized model.

    If the --from-file/-f option is given, TEMPLATE must be a path to a JSON
    template file. Otherwise, TEMPLATE must be the name of a template in the
    Apollo database.

    If the -o option is given, the trained model is serialized to the given
    destination. Otherwise the model is saved into the Apollo database.
    ''')


def parse_args(argv):
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=description(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-n',
        '--name',
        metavar='NAME',
        default=None,
        help='a name for the model (defaults to a UUID)',
    )

    parser.add_argument(
        '-f',
        '--from-file',
        action='store_true',
        help='load the template from a file',
    )

    parser.add_argument(
        '-o',
        metavar='DEST',
        dest='dest',
        default=None,
        help='output path for the model (defaults to the Apollo database)',
    )

    parser.add_argument(
        'template',
        metavar='TEMPLATE',
        help='the model template',
    )

    parser.add_argument(
        'data_file',
        metavar='DATA',
        nargs='?',
        default=sys.stdin,
        help='the training data (defaults to stdin)'
    )

    return parser.parse_args(argv)


def log(message):
    import logging
    logger = logging.getLogger(__name__)
    logger.info(message)


def make_model(args):
    import sys
    import apollo
    from apollo import models

    log('instantiating model from template')

    try:
        if args.file:
            return models.make_model_from(args.template, name=args.name)
        else:
            return models.make_model(args.template, name=args.name)
    except FileNotFoundError:
        print(
            f'No such template: {args.template}\n' +
            f'Hint: You can list known templates with `apollo ls templates`',
            file=sys.stderr,
        )
        sys.exit(1)


def read_targets(args):
    import pandas as pd
    log('reading training data')
    return pd.read_csv(args.data_file, parse_dates=True, index_col=0)


def fit(model, targets):
    log(f'training model with name {model.name}')
    model.fit(targets)


def main(argv):
    args = parse_args(argv)
    model = make_model(args)
    targets = read_targets(args)
    fit(model, targets)
    path = model.save(args.dest)
    print(path)
