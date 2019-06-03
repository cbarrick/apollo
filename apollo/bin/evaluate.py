import argparse
import logging
import math

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
from sklearn.model_selection import TimeSeriesSplit, PredefinedSplit

from apollo import models, casts
from apollo.models.base import list_trained_models
from apollo.models.base import load as load_model


def rmse(y_true, y_pred, **kwargs):
    '''A root-mean-squared error metric.

    Arguments:
        y_true (array):
            The true values.
        y_pred (array):
            The predicted values.
        **kwargs:
            Forwarded to :func:`sklearn.metrics.mean_squared_error`.
    '''
    return np.sqrt(mse(y_true, y_pred, **kwargs))


def make_splitter(args, first, last):
    '''Creates a splitter object for the evaluation.

    Arguments:
        args: The parsed CLI arguments.
        first: The first reftime in the data.
        last: The last reftime in the data.

    Returns:
        splitter: An object implementing the Scikit-learn splitter API.
    '''
    if args.cross_val is not None:
        splitter = TimeSeriesSplit(n_splits=args.cross_val)

    elif args.split is not None:
        test_pct = max(0, min(args.split, 1))

        # Total reftimes in the selected dataset.
        # FIXME: This is not correct in the presence of missing data.
        reftime_count = (last - first) // pd.Timedelta(6, 'h')

        testing_count = math.floor(reftime_count * test_pct)
        training_count = reftime_count - testing_count
        test_fold = np.concatenate((
            np.ones(training_count) * -1,   # -1 indicates training set
            np.zeros(testing_count)         # 0 indicates testing set, 1st fold
        ))

        splitter = PredefinedSplit(test_fold)

    else:
        # No mode was specified. This is currently unreachable.
        raise RuntimeError('unable to pick an evaluation mode')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Evaluate a trained Apollo model.',
    )

    parser.add_argument(
        'model',
        metavar='MODEL',
        type=str,
        help='the name of the saved model to be evaluated',
    )

    parser.add_argument(
        '-a',
        '--average',
        action='store_true',
        help='evaluate the mean error of forecasts for all hours',
    )

    parser.add_argument(
        '-c',
        '--csv',
        action='store_true',
        help='output the results as a csv',
    )

    selectors = parser.add_mutually_exclusive_group()

    selectors.add_argument(
        '-r',
        '--range',
        nargs=2,
        metavar=('START', 'STOP'),
        default=('2018-01-01T00:00', '2018-12-31T18:00'),
        help='evaluate using forecast on this range, inclusive'
    )

    # TODO: Add more selectors to be consistent with other Apollo CLIs.
    # This requires our models to be more Scikit-learn compatible (#65).
    # With multiple selectors, we can't set a default for argparse.

    modes = parser.add_mutually_exclusive_group(required=True)

    modes.add_argument(
        '-k',
        '--cross-val',
        metavar='K',
        type=int,
        help='evaluate using K-fold timeseries cross-validation',
    )

    modes.add_argument(
        '-p',
        '--split',
        metavar='RATIO',
        type=float,
        help='evaluate using a test-train split with this ratio',
    )

    args = parser.parse_args(argv)

    logging.info('Loading model...')
    model = load_model(args.model)

    first = casts.utc_timestamp(args.range[0]).floor('6h')
    last = casts.utc_timestamp(args.range[1]).floor('6h')

    splitter = make_splitter(args, first, last)

    logging.info('Evaluating...')
    results = model.validate(
        first=first,
        last=last,
        splitter=splitter,
        metrics=(mae, mse, rmse, r2),
        multioutput='uniform_average' if args.average else 'raw_values',
    )

    # TODO: We should make the validate method return a dataframe.
    results_df = pd.DataFrame(results, model.target_hours)

    if args.csv:
        print(results_df.to_csv())
    else:
        print('Results:')
        print(results_df.to_string())


if __name__ == '__main__':
    main()
