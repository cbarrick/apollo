import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import dask
from dask import array as da
from dask import distributed
from dask_ml.model_selection import GridSearchCV
from dask_ml.xgboost import XGBRegressor

import sklearn
from sklearn.model_selection import KFold, ParameterGrid

import apollo
from apollo.datasets import SolarDataset


logger = logging.getLogger(__name__)


#
# --------------------------------------------------

def now():
    return np.datetime64('now')


class WallTimer:
    def __init__(self):
        self.start = now()

    def __str__(self):
        return str(self.since())

    def since(self):
        return now() - self.start

    @classmethod
    def time(cls, fn, *args, **kwargs):
        timer = cls()
        result = fn(*args, **kwargs)
        print('Wall time:', timer)
        return result


#
# --------------------------------------------------

class Experiment:
    known_experiments = {}

    def __init__(self, main, name=None):
        if name is None:
            name = main.__name__

        self.main = main
        self.logger = logging.getLogger(str(name))
        self.known_experiments[name] = self

    def __call__(self):
        return self.main(self)

    @classmethod
    def run(cls, name, client=None):
        '''Lookup and execute an experiment by name, and save results to disk.

        Arguments:
            name (str):
                The name of the experiment.
            client (dask.distributed.Client):
                If a client is given, the job is submitted to it and a future
                is returned.

        Returns:
            pandas.DataFrame or Future[pandas.DataFrame]:
                The cross validation results.
        '''
        if client is not None:
            return client.submit(cls.run, name, client=None)

        exp = cls.known_experiments[name]
        retults = exp()

        timestamp = now()
        path = Path(f'results/{self.name}-{timestamp}.csv')
        logger.info(f'writing results to {path}')
        path.parent.mkdir(exist_ok=True)
        results.to_csv(path)

        return results

    def load_data(self, start='2017-01-01 00:00', stop='2017-12-31 18:00', **kwargs):
        '''Load the solar dataset as a pair of tabular dask arrays.

        Arguments:
            start (str):
                The timestamp of the first observation to be used.
            stop (str):
                The timestamp of the first observation to be used.
            **kwargs:
                Forwarded to the :class:`SolarDataset` constructor.

        Returns:
            Tuple[Array, Array]:
                A pair of 2D dask arrays ``(x, y)``.
        '''
        self.logger.info('loading data')
        timer = WallTimer()
        dataset = SolarDataset(start=start, stop=stop, **kwargs)
        x, y = dataset.tabular()
        self.logger.info(f'data loaded ({timer})')
        return x, y

    def cross_validate(self, x, y):
        '''Cross validate an :class:`XGBRegressor` on the given data.

        Arguments:
            x (Array):
                The observation data.
            y (Array):
                The target data.

        Returns:
            pandas.DataFrame:
                The cross validation results.
        '''
        self.logger.info('cross validating model')
        timer = WallTimer()
        model = GridSearchCV(
            estimator = XGBRegressor(),
            param_grid = dict(n_estimators=[200], max_depth=[50]),
            cv = KFold(n_splits=3, shuffle=True),
            scoring = 'neg_mean_absolute_error',
            return_train_score = False,
            refit = False,
        ).fit(x, y)

        self.logger.info(f'cross validation complete ({timer})')
        results = pd.DataFrame(model.cv_results_)
        return results

    def trial(self, **kwargs):
        '''Select a dataset and cross validate an :class:`XGBRegressor`.

        This is essentially :meth:`Experiment.load_data` followed by
        :meth:`Experiment.cross_validate`. The kwargs used to select the
        dataset are attached to the results.

        Arguments:
            **kwargs:
                Forwarded to :meth:`Experiment.load_data` and used to select
                the training data.

        Returns:
            pandas.DataFrame:
                The cross validation results.
        '''
        self.logger.info('starting trial: {kwargs}')
        x, y = self.load_data(**kwargs)
        results = self.cross_validate(x, y)

        for k, v in kwargs.items():
            results[k] = [v]

        self.logger.info('trial complete: {kwargs}')
        return results

    def search(self, **kwargs):
        '''Perform a :meth:`trial` for each dataset configuration in a grid.

        Arguments:
            **kwargs:
                A :class:`sklearn.model_selection.ParameterGrid` over the
                dataset configurations to search.

        Returns:
            pandas.DataFrame:
                The cross validation results.
        '''
        param_grid = ParameterGrid(kwargs)
        results = [self.trial(**params) for params in param_grid]
        results = pd.concat(results)
        return results


#
# --------------------------------------------------

@Experiment
def sanity_check(exp):
    return exp.trial(lag=4, forecast=6)


@Experiment
def lag_vs_forecast(exp):
    return exp.search(
        lag = [0, 1, 2, 4, 8],
        forecast = [0, 1, 2, 4, 8, 16, 32],
    )


#
# --------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level='DEBUG',
        format='{levelname:.3} [{name}] {message}',
        style='{',
    )


def main():
    parser = ArgumentParser('Run an experiment')
    parser.add_argument('exp', metavar='EXP', help='an experiment to run')
    parser.add_argument('--dask', metavar='SCHEDULER', default=None, help='the dask scheduler address')
    args = parser.parse_args()

    client = distributed.Client(args.dask or '127.0.0.1:8786')

    setup_logging()
    client.run(setup_logging)

    name = args.exp
    Experiment.run(name)


if __name__ == '__main__':
    main()
