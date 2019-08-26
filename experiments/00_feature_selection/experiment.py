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


TEMPLATE = (
    ('estimator', "sklearn.linear_model.LinearRegression"),
    # ('features', ("DSWRF_SFC",)),
    # ('add_time_of_day', False),
    # ('add_time_of_year', False),
    ('daylight_only', True),
    ('standardize', True),
    ('shape', 12000),
)


def log(message):
    import logging
    logger = logging.getLogger(__name__)
    logger.info(message)


def greedy_search(train, test, features=FEATURES, template=TEMPLATE):
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
        template (dict):
            The base model template for the search.

    Returns:
        best_model (object):
            The best model.
        best_features (list of str):
            The features selected by the search.
        best_template (dict):
            The template of the best model.
    '''
    import apollo
    from apollo import metrics, models

    best_features = []
    best_score = float('-inf')
    best_template = None
    best_model = None

    while True:
        log(f'Current best features: {best_features}')
        prev_features = best_features.copy()

        for f in features:
            # Skip features that have already been selected.
            if f in prev_features: continue

            # Determine the feature set to use for this iteration.
            current_features = prev_features + [f]
            log(f'Checking {current_features}')

            # Create the template for this iteration.
            # The '_time_of_day' and '_time_of_year' features are special.
            # They come from a preprocess step rather than the raw NAM data.
            tmpl_features = current_features.copy()
            try:
                tmpl_features.remove('_time_of_day')
                time_of_day = True
            except ValueError:
                time_of_day = False
            try:
                tmpl_features.remove('_time_of_day')
                time_of_year = True
            except ValueError:
                time_of_year = False
            tmpl = dict(
                **template,
                features=tmpl_features,
                add_time_of_day=time_of_day,
                add_time_of_year=time_of_year,
            )

            # Evaluate the template.
            # We sum the r^2 for all columns into a scalar score, though
            # technically r^2 shouldn't be compared for different targets.
            model = models.make_model_from(tmpl)
            model.fit(train)
            predictions = model.predict(test.index)
            score = metrics.r2(test, predictions).sum()

            # Update our "best" variables if this score is better.
            if best_score < score:
                best_features = current_features
                best_score = score
                best_template = tmpl
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
        level='DEBUG',
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
