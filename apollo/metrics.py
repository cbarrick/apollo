from sklearn.metrics import r2_score


def _apply(fn, targets, predictions, **kwargs):
    '''Apply a metric over the targets and predictions.

    The targets and predictions are data frames with the same columns and index.
    The columns may differ in order, but the row index must be the same for
    both.

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
    # Reindex to ensure the columns appear in the same order.
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
    scores = _apply(r2_score, targets, predictions)
    scores.name = 'r2'
    return scores
