import argparse
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score as r2

from apollo.models.base import list_trained_models
from apollo.models.base import load as load_model
from apollo.models import *
from apollo.validation import cross_validate, split_validate


def rmse(y_true, y_pred):
    return mse(y_true, y_pred)**0.5


def main():
    model_names = list_trained_models()
    parser = argparse.ArgumentParser(
        description='Apollo Model Evaluator',
        argument_default=argparse.SUPPRESS,
    )
    # specify the model to eval
    parser.add_argument('name', type=str, choices=model_names,
                        help='The name of the saved model to be evaluated.')

    parser.add_argument('mode', type=str, choices=('cross_val', 'split'),
                        help='Validation mode.  k-fold timeseries cross-validation or train-test split.')

    parser.add_argument('--start', '-b', default='2017-01-01 00:00', type=str,
                        help='The first reftime in the dataset to be used for training.  '
                             'Any string accepted by numpy\'s datetime64 constructor will work.')

    parser.add_argument('--stop', '-e', default='2017-12-31 18:00', type=str,
                        help='The final reftime in the dataset to be used for training. '
                             'Any string accepted by numpy\'s datetime64 constructor will work.')

    parser.add_argument('--k', '-k', default=5, type=int,
                        help='Number of folds to use for cross-validation.  Ignored if using `split` mode.')

    parser.add_argument('--split_size', '-p', default=0.25, type=float,
                        help='Proportion of the dataset to be used for testing.  '
                             'Ignored if using `cross_val` mode.')


    # parse args
    args = parser.parse_args()
    args = vars(args)

    args['split_size'] = max(0, min(args['split_size'], 1))  # squeeze training size between [0, 1]

    print('Loading model...')
    model_name = args['name']
    model = load_model(model_name)

    print('Evaluating...')
    if args['mode'] == 'split':
        results = split_validate(model, first=args['start'], last=args['stop'], test_size=args['split_size'],
                                 metrics=(mae, mse, rmse, r2))
    else:
        results = cross_validate(model, first=args['start'], last=args['stop'], k=args['k'],
                                 metrics=(mae, mse, rmse, r2))

    print('Results:\n%s' % results)


if __name__ == '__main__':
    main()
