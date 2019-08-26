FEATURES = (
    'PRES_SFC',
    'HGT_SFC',
    'HGT_TOA',
    'TMP_SFC',
    'VIS_SFC',
    'DSWRF_SFC',
    'DLWRF_SFC',
    'TCC_EATM',
    'UGRD_TOA',
    'VGRD_TOA',
    '_time_of_day',
    '_time_of_year',
)


BASE_TEMPLATE = (
    ('estimator', "sklearn.linear_model.LinearRegression"),
    ('daylight_only', True),
    ('standardize', True),
    ('shape', 12000),
)


def log(message):
    import logging
    logger = logging.getLogger(__name__)
    logger.info(message)


def fit_and_eval(base_template, features, train, test):
    '''Create a model from a template, train it, then score it.

    Arguments:
        base_template (dict):
            The base template for the model.
        features (list of str):
            The features to use. The features `'_time_of_day'` and
            `'_time_of_year'` are special; they correspond the their
            associated preprocess steps rather than direct features.
        train (pandas.DataFrame):
            The training data.
        test (pandas.DataFrame):
            The test data.

    Returns:
        template (dict):
            The actual template of the model.
        features:
            The feature list passed in.
        model (object):
            The trained model.
        score (float):
            The r^2 coefficient of determination.
            See :func:`apollo.metrics.r2`.
    '''
    import apollo
    from apollo import metrics, models

    # Create the actual template.
    # The '_time_of_day' and '_time_of_year' features are special.
    # They come from a preprocess step rather than the raw NAM data.
    try:
        features.remove('_time_of_day')
        time_of_day = True
    except ValueError:
        time_of_day = False
    try:
        features.remove('_time_of_day')
        time_of_year = True
    except ValueError:
        time_of_year = False
    template = dict(base_template)
    template.update(
        features=features,
        add_time_of_day=time_of_day,
        add_time_of_year=time_of_year,
    )

    # We sum the r^2 for all columns into a scalar score, though
    # technically r^2 shouldn't be compared for different targets.
    model = models.make_model_from(template)
    model.fit(train)
    predictions = model.predict(test.index)
    score = metrics.r2(test, predictions).sum()

    return template, features, model, score


def greedy_search(train, test, features=FEATURES, base_template=BASE_TEMPLATE):
    '''Select a feature set using a greedy search.

    Starting from an empty list, this function builds a feature set by
    iterativly adding the feature to the list which improves the r^2 most. The
    search ends when no feature is found which causes an improvement.

    Arguments:
        train (pandas.DataFrame):
            The train set.
        test (pandas.DataFrame):
            The test set.
        features (list of str):
            The names of features to consider in the search. The features
            `'_time_of_day'` and `'_time_of_year'` are special; they correspond
            the their associated preprocess steps rather than direct features.
        base_template (dict):
            The base model template for the search.

    Returns:
        best_model (object):
            The best model.
        best_features (list of str):
            The features selected by the search.
        best_template (dict):
            The template of the best model.
    '''
    from multiprocessing import Pool
    pool = Pool()

    best_features = []
    best_score = float('-inf')
    best_template = None
    best_model = None

    while True:
        log(f'Current best features: {best_features}')
        prev_features = best_features.copy()

        tasks = []

        # Spawn a training process for each feature.
        for f in features:
            # Skip features that have already been selected.
            if f in prev_features: continue

            # Determine the feature set to use for this iteration.
            task_features = prev_features + [f]

            # Spawn the task.
            log(f'Spawning task: {task_features}')
            args = [base_template, task_features, train, test]
            task = pool.apply_async(fit_and_eval, args)
            tasks.append(task)

        # Update our "best" variables if any score is better.
        for task in tasks:
            task_template, task_features, model, score = task.get()
            if best_score < score:
                log(f'Found improvement: {task_features}')
                best_features = task_features
                best_score = score
                best_template = task_template
                best_model = model

        # After trying all variable, if the feature set has not changed,
        # then no feature improved the score. We can terminate the search.
        if best_features == prev_features:
            break

    # Return best model, feature set, and template.
    return best_model, best_features, best_template


if __name__ == '__main__':
    import logging

    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level='INFO',
    )

    import json
    import pandas as pd
    from pathlib import Path

    root = Path(__file__).parent

    log('Loading train.csv')
    train = pd.read_csv(root / 'train.csv', parse_dates=True, index_col=0)

    log('Loading test.csv')
    test = pd.read_csv(root / 'test.csv', parse_dates=True, index_col=0)

    log('Starting search')
    best_model, best_features, best_template = greedy_search(train, test)

    log(f'Best features: {best_features}')

    log('Writing model')
    best_model.save(root / 'MODEL.model')

    log('Writing template')
    with (root / 'MODEL.json').open('w') as fd:
        json.dump(best_template, fd)
