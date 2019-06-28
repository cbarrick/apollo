import argparse
import logging
import sys

import pandas as pd

from apollo import models2
from apollo.datasets import ga_power


logger = logging.getLogger(__name__)


# TODO: Should we move this to `apollo.datasets.ga_power`?
def load_targets(times, features):
    '''Load targets for the given times.

    Arguments:
        times (np.ndarray like):
            A series of timestamps.

    Returns:
        pandas.DataFrame:
            A data fram indexed by the observation time.
    '''
    # Load all data in the given time range.
    times = pd.DatetimeIndex(times, name='time').unique()
    lo = times.min().floor('1h')
    hi = times.max().ceil('1h')
    targets = ga_power.open(lo, hi)

    # Localize the index (tz-naive timestamps to UTC timestamps).
    targets.index = targets.index.tz_localize('UTC')

    # Select only the desired features.
    targets = targets[list(features)]

    # Cast to single precision and drop NaNs and infinities.
    targets = targets.astype('float32')
    targets = targets.replace([np.inf, -np.inf], np.nan)
    targets = targets.dropna()

    # Average by hour then filter to the given times.
    hour = targets.index.floor('1h')
    targets = targets.groupby(hour).mean()
    index = targets.index.intersection(times)
    targets = targets.loc[index]
    return targets


def main(argv):
    parser = argparse.ArgumentParser(
        description='train a new model'
    )

    args = parser.parse_args(argv)

    # DEBUG:
    sys.exit(0)

    logger.info('loading targets')
    train_times = pd.date_range(
        start='2018-01-01 00:00:00',
        end='2018-12-31 18:00:00',
        freq='h',
        tz='UTC',
    )
    targets = load_targets(train_times, ['UGABPOA1IRR'])

    logger.info('training the model')
    model = models2.Model(
        estimator='apollo.models2.estimators.linear',
        features=['DSWRF_SFC'],
        temporal_features=False,
        shape=12000,
    )
    model.fit(targets)

    logger.info('generating predictions')
    test_times = pd.date_range(
        '2018-01-01 00:00:00',
        '2018-12-31 18:00:00',
        freq='h',
        tz='UTC',
    )
    predictions = model.predict(test_times)
    print(predictions.to_csv())
