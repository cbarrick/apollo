import argparse

from apollo.models.base import list_trained_models
from apollo.models.base import delete as delete_model


def main():
    model_names = list_trained_models()
    parser = argparse.ArgumentParser(
        description='Apollo Model Deleter',
        argument_default=argparse.SUPPRESS,
    )
    # specify the type of model and give it a name
    parser.add_argument('name', type=str, choices=model_names,
                        help='The name of the saved model you wish to delete')

    # parse args
    args = parser.parse_args()
    args = vars(args)

    delete_model(args['name'])


if __name__ == '__main__':
    main()
