import argparse
import logging

from apollo import timestamps
from apollo.models.base import save as save_model
from apollo.models.base import list_known_models


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Train a new Apollo model.',
    )

    parser.add_argument(
        'model',
        metavar='MODEL',
        type=str,
        help='the class of the model to train',
    )

    parser.add_argument(
        '--set',
        metavar='KEY=VALUE',
        type=str,
        action='append',
        default=[],
        dest='kwrags',  # Note that this is parsed into ``args.kwargs``.
        help='set a hyper-parameter for the model, may be specified multiple times',
    )

    selectors = parser.add_mutually_exclusive_group()

    selectors.add_argument(
        '-r',
        '--range',
        nargs=2,
        metavar=('START', 'STOP'),
        default=('2017-01-01T00:00', '2017-12-31T18:00'),
        help='train on all forecast on this range, inclusive'
    )

    # TODO: Add more selectors to be consistent with other Apollo CLIs.
    # This requires our models to be more Scikit-learn compatible (#65).
    # With multiple selectors, we can't set a default for argparse.

    args = parser.parse_args(argv)

    classes = {model.__name__: model for model in list_known_models()}

    ModelClass = classes[args.model]
    kwargs = dict(pair.split('=') for pair in args.kwargs)
    first = timestamps.utc_timestamp(args.range[0]).floor('6h')
    last = timestamps.utc_timestamp(args.range[1]).floor('6h')

    logging.info(f'Constructing {args.model} with kwargs: {kwargs}')
    model = ModelClass(**kwargs)
    model.fit(first, last)

    logging.info(f'Saving model with name "{model.name}"')
    save_model(model)


if __name__ == '__main__':
    main()
