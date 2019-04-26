import argparse

from apollo.models.base import list_trained_models
from apollo.models.base import delete as delete_model


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Delete a trained model.',
    )

    parser.add_argument(
        'model',
        type=str,
        metavar='MODEL',
        help='the name of the model',
    )

    args = parser.parse_args(argv)

    delete_model(args.model)


if __name__ == '__main__':
    main(argv)
