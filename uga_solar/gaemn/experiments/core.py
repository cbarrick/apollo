import logging
import math
import re
import signal
import sys

from collections import OrderedDict

import numpy as np
import scipy as sp
import tensorflow as tf

from scipy import stats
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error

from uga_solar.gaemn.data import gaemn15

logger = logging.getLogger(__name__)


def setup():
    '''Parses CLI arguments and sets global configuration.'''
    import argparse

    parser = argparse.ArgumentParser(description='Perform the experiment')
    parser.add_argument('--log', default='WARNING', type=str, help='the level of logging details')
    parser.add_argument('--seed', default=1337, type=int, help='the random seed of the experiment')
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.log))
    logging.basicConfig(level=numeric_level)

    np.random.seed(args.seed)
    tf.set_random_seed(np.random.randint(0xffffffff))

    def sigint_handler(signal, frame):
        print()
        logger.warn('Manual halt (SIGINT)')
        sys.exit()

    signal.signal(signal.SIGINT, sigint_handler)


def percent_split(
        estimators,
        datasets,
        split=0.8,
        nfolds=10,
        metric=mean_absolute_error,
        desc=False,
):
    results = Results()
    for dataset_class, datasets_grid in datasets.items():
        for dataset_params in ParameterGrid(datasets_grid):
            for est_class, est_grid in estimators.items():
                for est_params in ParameterGrid(est_grid):
                    estimator = est_class(**est_params)
                    train, test = dataset_class(**dataset_params).split(split)
                    fit(estimator, train, desc)
                    scores = score(estimator, test, nfolds, metric)
                    results.record(scores, estimator, train)
    return results


def cross_site_percent_split(
        estimators,
        datasets,
        split=0.8,
        nfolds=10,
        metric=mean_absolute_error,
        desc=False,
):
    results = Results()
    for dataset_class, datasets_grid in datasets.items():
        for dataset_params in ParameterGrid(datasets_grid):
            for est_class, est_grid in estimators.items():
                for est_params in ParameterGrid(est_grid):
                    estimator = est_class(**est_params)
                    train, _ = dataset_class(**dataset_params).split(split)
                    fit(estimator, train, desc)
            for dataset_class, datasets_grid in datasets.items():
                for dataset_params in ParameterGrid(datasets_grid):
                    _, test = dataset_class(**dataset_params).split(split)
                    scores = score(estimator, test, nfolds, metric)
                    results.record(scores, estimator, train, test)
    return results


def train_test(
        estimators,
        train_set,
        test_sets,
        nfolds=10,
        metric=mean_absolute_error,
        desc=False,
):
    results = Results()
    for est_class, est_grid in estimators.items():
        for est_params in ParameterGrid(est_grid):
            estimator = est_class(**est_params)
            fit(estimator, train_set, desc)
            for dataset_class, datasets_grid in test_sets.items():
                for dataset_params in ParameterGrid(datasets_grid):
                    test = dataset_class(**dataset_params)
                    scores = score(estimator, test, nfolds, metric)
                    results.record(scores, estimator, test.city)
    return results


def fit(estimator, dataset, desc):
    logger.info('fitting {} to {}'.format(estimator, dataset))
    if hasattr(estimator, 'partial_fit'):
        sgd(estimator, dataset, desc)
    else:
        bgd(estimator, dataset)


def bgd(estimator, train):
    estimator.fit(train.data, train.target)


def sgd(estimator, train, desc, val_split=0.9, max_epochs=1000, patience=20, batch_size=32):
    train, val = train.split(val_split)
    t = patience
    best = math.inf if not desc else -math.inf
    for epoch in range(max_epochs):
        for x, y in train.batch():
            estimator.partial_fit(x, y)
        pred = estimator.predict(val.data)
        score = metric(val.target, pred)
        logger.info('epoch={}, score={}'.format(epoch, score))
        if (desc and score < best) or best < score:
            t -= 1
            if t == 0:
                break
        else:
            t = patience
            best = score
    # TODO: rewind to best model
    return


def score(estimator, dataset, nfolds, metric):
    logger.info('scoring {} on {}'.format(estimator, dataset))
    n = len(dataset.data) // nfolds
    results = []
    for i in range(nfolds):
        s = slice(n * i, n * (i + 1))
        pred = estimator.predict(dataset.data[s])
        score = metric(dataset.target[s], pred)
        results.append(score)
    logger.debug(results)
    return results


class Results:
    space = re.compile('\s+')

    def __init__(self, desc=False):
        self.desc = desc
        self._trials = {}
        self.nkeys = -1

    def __str__(self):
        return self.summary()

    def record(self, scores, *keys):
        # Convert keys to strings.
        # This avoids a memory leak:
        # we don't want to keep references to datasets!
        keys = tuple(Results.space.sub(' ', str(k)) for k in keys)
        l = self._trials.setdefault(keys, [])
        l += scores

    def trials(self, level=None):
        flat = {}
        for k, v in self._trials.items():
            l = flat.setdefault(k[:level], [])
            l += v
        flat = OrderedDict(sorted(flat.items(), key=lambda i: np.mean(i[1])))
        return flat

    def summary(self, level=None):
        trials = self.trials(level)
        ttest = self._ttest(trials)
        str = ''
        str += 'METRIC  TRIAL\n'
        str += '------------------------------------------------------------------------\n'
        for key, scores in trials.items():
            str += '{:<7.3f} {}\n'.format(np.mean(scores), key[0])
            for k in key[1:]:
                str += ' ' * 8 + '{}\n'.format(k)
            str += '\n'
        str += '\n'
        str += 't-Test Matrix (p-values)\n'
        str += '------------------------------------------------------------------------\n'
        for i, row in enumerate(ttest):
            for j, val in enumerate(row):
                if i == j:
                    str += '   --    '
                else:
                    str += '{:8.3%} '.format(val)
            str += '\n'
        return str

    def ttest(self, level=None):
        trials = self.trials(level)
        return _ttest(trials)

    def _ttest(self, trials):
        n = len(trials)
        t_mat = np.zeros((n, n))
        for i, a_scores in enumerate(trials.values()):
            for j, b_scores in enumerate(trials.values()):
                if i == j:
                    continue
                t, p = sp.stats.ttest_rel(a_scores, b_scores)
                t_mat[i, j] = p
        return t_mat
