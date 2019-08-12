import numpy as np
import pandas as pd

import sklearn
import sklearn.metrics as skmetrics


def _apply(fn, targets, predictions, **kwargs):
    '''Apply a metric over the targets and predictions.

    Arguments:
        fn (callable):
            The metric function.
        targets (pandas.DataFrame):
            The true observations.
        predictions (pandas.DataFrame):
            The predicted values.
        kwargs:
            Additional keyword arguments are passed to the metric function.

    Returns:
        pandas.Series:
            The metric computed for each column.
    '''
    # Reindex columns to ensure they appear in the same order.
    predictions = predictions.reindex(targets.columns, axis=1)
    columns = targets.columns

    # Compute the metric.
    targets = targets.to_numpy()
    predictions = predictions.to_numpy()
    scores = fn(targets, predictions, **kwargs)
    return pd.Series(scores, index=columns)


def r2(targets, predictions):
    '''Compute the coefficient of determination.

    This function uses the same formula as Scikit-learn:
    <https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score>

    Arguments:
        targets (pandas.DataFrame):
            The true observations.
        predictions (pandas.DataFrame):
            The predicted values.

    Returns:
        pandas.Series:
            The metric computed for each column.
    '''
    scores = _apply(skmetrics.r2_score, targets, predictions)
    scores.name = 'r2'
    return scores


def mae(targets, predictions):
    '''Compute the mean absolute error.

    Arguments:
        targets (pandas.DataFrame):
            The true observations.
        predictions (pandas.DataFrame):
            The predicted values.

    Returns:
        pandas.Series:
            The metric computed for each column.
    '''
    scores = _apply(skmetrics.mean_absolute_error, targets, predictions)
    scores.name = 'mae'
    return scores


def rmse(targets, predictions):
    '''Compute the root mean squared error.

    Arguments:
        targets (pandas.DataFrame):
            The true observations.
        predictions (pandas.DataFrame):
            The predicted values.

    Returns:
        pandas.Series:
            The metric computed for each column.
    '''
    scores = _apply(skmetrics.mean_squared_error, targets, predictions)
    scores = np.sqrt(scores)
    scores.name = 'rmse'
    return scores


def all(targets, predictions):
    '''Compute all available metrics.

    Arguments:
        targets (pandas.DataFrame):
            The true observations.
        predictions (pandas.DataFrame):
            The predicted values.

    Returns:
        pandas.DataFrame:
            A data frame with the same columns as the targets/predictions and
            one row for each metric.
    '''
    return pd.DataFrame({
        'r2': r2(targets, predictions),
        'mae': mae(targets, predictions),
        'rmse': rmse(targets, predictions),
    })
