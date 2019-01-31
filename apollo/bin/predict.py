import argparse
import pandas as pd

import apollo.datasets.nam as nam
from apollo.models.base import list_trained_models
from apollo.models.base import load as load_model
from apollo.models import *


def main():
    model_names = list_trained_models()
    parser = argparse.ArgumentParser(
        description='Apollo Machine Learning Model Trainer',
        argument_default=argparse.SUPPRESS,
    )
    # specify the type of model and give it a name
    parser.add_argument('name', type=str, choices=model_names,
                        help='The name of the saved model used to make prediction.')

    parser.add_argument('--reftime', '-r', default='2018-01-01 00:00', type=str,
                        help='The reftime for which predictions should be made.  '
                             'Any string accepted by numpy\'s datetime64 constructor will work.  '
                             'Ignored if the `latest` flag is set.')

    parser.add_argument('--latest', '-l', action='store_true',
                        help='If set, a prediction will be generated for the past reftime which is closest to the '
                             'current datetime.')

    # parse args
    args = parser.parse_args()
    args = vars(args)

    reftime = pd.Timestamp(args['reftime'])
    if 'latest' in args:
        reftime = pd.Timestamp('now').floor('6h')

    # ensure data for the requested reftime is cached
    print(f'Caching NAM data for reftime {reftime}...')
    nam.open(reftime - pd.Timedelta(6, 'h'), reftime)
    print('Done.')

    model = load_model(args['name'])
    forecast = model.forecast(reftime)

    # TODO - write predictions to a file


if __name__ == '__main__':
    main()
