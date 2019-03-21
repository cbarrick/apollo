import logging
import sys

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

import apollo.datasets.ga_power as ga_power
from apollo.datasets.nam import CacheMiss
from apollo.models.base import ValidatableModel

logger = logging.getLogger(__name__)


def cross_validate(model, first, last, metrics=(mean_absolute_error,), k=3):
    ''' Evaluate a model using k-fold cross-validation

    Args:
        model (ValidatableModel):
            The model to evaluate
        first (str or pd.Timestamp):
            The timestamp of the first data point to use for cross-validation
        last (str or pd.Timestamp):
            The timestamp of the last data point to use for cross-validation
        metrics (Iterable[Callable]):
            Set of evaluation metrics to apply.  Each metric should have
            a signature like metric_name(y_true, y_predicted)
        k (int):
            Number of folds to use

    Returns:
        DataFrame:
            A dataframe indexed by target hour with a column for each metric

    '''

    assert isinstance(model, ValidatableModel)

    # records has the following structure, where the key is the forecast hour
    # {
    #     1: {
    #         'y_true': [],
    #         'y_pred': []
    #     },
    #     2: {
    #         'y_true': [],
    #         'y_pred': []
    #     },
    #     . . .
    # }
    records = {i: {'y_true': list(), 'y_pred': list()}
               for i in model.target_hours}

    first = pd.Timestamp(first).round(freq='6h')
    last = pd.Timestamp(last).round(freq='6h')
    reftimes = pd.date_range(first, last, freq='6h')

    true_values = ga_power.open_sqlite(model.target, start=first,
        stop=(last + pd.Timedelta(48, 'h'))).to_dataframe()

    true_values.rename(
        columns={true_values.columns[0]: 'true_val'},
        inplace=True)

    time_series_splitter = TimeSeriesSplit(n_splits=k)

    for train_index, test_index in time_series_splitter.split(reftimes):
        train_reftimes = reftimes[train_index]
        test_reftimes = reftimes[test_index]

        # train the model using the training set
        model.fit(train_reftimes[0], train_reftimes[-1])

        # make predictions for each reftime in the testing set
        for reftime in test_reftimes:
            try:
                predictions = model.forecast(reftime)
                # predictions will be a DataFrame
                # of (reftime, target) pairs for each target hour
                predictions.rename(
                    columns={predictions.columns[0]: 'predicted'}, inplace=True)

                # match true values and find difference
                matched = pd.concat([predictions, true_values],
                                    axis=1, join='inner')

                for timestamp, vals in matched.iterrows():
                    hour = (timestamp - reftime) // pd.Timedelta(1, 'h')
                    if not hour in records:
                        records[hour] = {'y_true': list(), 'y_pred': list()}

                    records[hour]['y_pred'].append(vals['predicted'])
                    records[hour]['y_true'].append(vals['true_val'])

            # if anything goes wrong, omit the results from the error estimation
            except CacheMiss:
                logger.warning(f'Omitting results for reftime {reftime}')
                logger.error(sys.exc_info()[0])
                pass

    # compute error using each of the metrics
    results = pd.DataFrame(index=model.target_hours,
                           columns=[metric.__name__ for metric in metrics])
    for hour in records:
        y_true, y_pred = records[hour]['y_true'], records[hour]['y_pred']
        for metric in metrics:
            error = metric(y_true, y_pred)
            results.loc[hour, metric.__name__] = error

    return results


def split_validate(model, first, last, test_size=0.25,
                   metrics=(mean_absolute_error,), ):
    ''' Evaluate a model using a train-test split

    Args:
        model (ValidatableModel):
            The model to evaluate
        first (str or pd.Timestamp):
            The timestamp of the first data point to use for cross-validation
        last (str or pd.Timestamp):
            The timestamp of the last data point to use for cross-validation
        metrics (Iterable[Callable]):
            Set of evaluation metrics to apply. Each metric should have
            a signature like metric_name(y_true, y_predicted)
        test_size (float):
            Proportion of the dataset to use for testing.
            The complement will be used for training.

    Returns:
        DataFrame:
            A dataframe indexed by target hour with a column for each metric

    '''

    assert isinstance(model, ValidatableModel)

    # records has the following structure, where the key is the forecast hour
    # {
    #     1: {
    #         'y_true': [],
    #         'y_pred': []
    #     },
    #     2: {
    #         'y_true': [],
    #         'y_pred': []
    #     },
    #     . . .
    # }
    records = {i: {'y_true': list(), 'y_pred': list()}
               for i in model.target_hours}

    first = pd.Timestamp(first).round(freq='6h')
    last = pd.Timestamp(last).round(freq='6h')
    reftimes = pd.date_range(first, last, freq='6h')

    true_value_stop = last + pd.Timedelta(48, 'h')
    true_values = ga_power.open_sqlite(model.target, start=first,
                                       stop=true_value_stop).to_dataframe()
    true_values.rename(
        columns={true_values.columns[0]: 'true_val'}, inplace=True)

    split_index = round(len(reftimes) * (1-test_size))
    train_reftimes = reftimes[0:split_index].tolist()
    test_reftimes = reftimes[split_index:].tolist()

    # train the model using the training set
    model.fit(train_reftimes[0], train_reftimes[-1])

    # make predictions for each reftime in the testing set
    for reftime in test_reftimes:
        try:
            predictions = model.forecast(reftime)
            # predictions will be a DataFrame of (reftime, target) pairs
            predictions.rename(
                columns={predictions.columns[0]: 'predicted'}, inplace=True)

            # match true values and find difference
            matched = pd.concat([predictions, true_values], axis=1, join='inner')

            for timestamp, vals in matched.iterrows():
                hour = (timestamp - reftime) // pd.Timedelta(1, 'h')
                if not hour in records:
                    records[hour] = {'y_true': list(), 'y_pred': list()}

                records[hour]['y_pred'].append(vals['predicted'])
                records[hour]['y_true'].append(vals['true_val'])
        # if anything goes wrong, omit the results from the error estimation
        except CacheMiss:
            logger.warning(f'Omitting validation results for reftime {reftime}')
            logger.error(sys.exc_info()[0])
            pass

    # compute error using each of the metrics
    results = pd.DataFrame(index=model.target_hours,
                           columns=[metric.__name__ for metric in metrics])
    for hour in records:
        y_true, y_pred = records[hour]['y_true'], records[hour]['y_pred']
        for metric in metrics:
            error = metric(y_true, y_pred)
            results.loc[hour, metric.__name__] = error

    return results
