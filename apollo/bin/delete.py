import argparse

from apollo.models.base import list_trained_models
from apollo.models.base import delete as delete_model


def main(argv=None):
    model_names = list_trained_models()

    parser = argparse.ArgumentParser(
        description='Apollo Model Deleter',
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument(
        'name',
        type=str,
        choices=model_names,
        help='the name of the saved model you wish to delete'
    )

    args = parser.parse_args(argv)
    args = vars(args)

    delete_model(args['name'])


if __name__ == '__main__':
    main(argv)
