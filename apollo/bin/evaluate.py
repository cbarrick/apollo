import argparse
from math import floor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as mae, \
    mean_squared_error as mse, r2_score as r2
from sklearn.model_selection import TimeSeriesSplit, PredefinedSplit

from apollo.models.base import list_trained_models
from apollo.models.base import load as load_model
from apollo.models import *


# define custom RMSE metric
def rmse(y_true, y_pred, **kwargs):
    return mse(y_true, y_pred, **kwargs)**0.5


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
                        help='Validation mode.'
                             ' K-fold timeseries cross-validation'
                             ' or train-test split.')

    parser.add_argument('--first', '-b', default='2017-01-01 00:00', type=str,
                        help='The first reftime in the dataset.')

    parser.add_argument('--last', '-e', default='2017-12-31 18:00', type=str,
                        help='The final reftime in the dataset.')

    parser.add_argument('--k', '-k', default=5, type=int,
                        help='Number of folds to use for cross-validation.'
                             ' Ignored if using `split` mode.')

    parser.add_argument('--split_size', '-p', default=0.25, type=float,
                        help='Proportion of the dataset to be used for testing.'
                             ' Ignored if using `cross_val` mode.')

    parser.add_argument('--average', '-a', action='store_true',
                        help='If set, the evaluations of each target hour will'
                             ' be reduced to a single value by taking the mean'
                             ' with uniform weights')

    # parse args
    args = parser.parse_args()
    args = vars(args)

    print('Loading model...')
    model_name = args['name']
    model = load_model(model_name)

    multioutput = 'raw_values'
    if 'average' in args:
        multioutput = 'uniform_average'

    print('Evaluating...')
    metrics = (mae, mse, rmse, r2)
    if args['mode'] == 'split':
        # use a time-series splitter
        splitter = TimeSeriesSplit(n_splits=args['k'])
    else:
        # create custom splitter with PredefinedSplit
        test_pct = max(0, min(args['split_size'], 1))
        first = pd.Timestamp(args['first']).floor(freq='6h')
        last = pd.Timestamp(args['last']).floor(freq='6h')
        # total reftimes in the selected dataset
        reftime_count = (last - first) // pd.Timedelta(6, 'h')

        # create index for PredefinedSplit
        testing_count = floor(reftime_count*test_pct)
        training_count = reftime_count - testing_count
        test_fold = np.concatenate((
            np.ones(training_count) * -1,   # -1 indicates training set
            np.zeros(testing_count)         # 0 indicates testing set, 1st fold
        ))
        splitter = PredefinedSplit(test_fold)

    results = model.validate(
        first=args['first'], last=args['last'],
        splitter=splitter, metrics=metrics,
        multioutput=multioutput)

    print('Results:\n%s' % results)


if __name__ == '__main__':
    main()
