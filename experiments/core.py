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

from data import gaemn15

logger = logging.getLogger(__name__)


def setup():
    '''Parses CLI arguments and sets global configuration.'''
    import argparse

    parser = argparse.ArgumentParser(description='Perform the experiment')
    parser.add_argument(
        '--log',
        default='WARNING',
        type=str,
        help='the level of logging details')
    parser.add_argument(
        '--seed',
        default=1337,
        type=int,
        help='the random seed of the experiment')
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


def cross_site(estimators,
               datasets,
               nfolds=10,
               metric=mean_absolute_error,
               desc=False):
    results = {}
    for dataset_class, datasets_grid in datasets.items():
        for dataset_params in ParameterGrid(datasets_grid):
            for est_class, est_grid in estimators.items():
                for est_params in ParameterGrid(est_grid):
                    estimator = est_class(**est_params)
                    train = dataset_class(**dataset_params)
                    train_repr = re.sub('\s+', ' ', train.__repr__())
                    estimator_repr = re.sub('\s+', ' ', estimator.__repr__())
                    fit(estimator, train, desc)
                    key = (train_repr, estimator_repr)
                    results[key] = []
            for dataset_class, datasets_grid in datasets.items():
                for dataset_params in ParameterGrid(datasets_grid):
                    test = dataset_class(**dataset_params)
                    scores = score(estimator, test, nfolds, metric)
                    results[key] += scores
    return Results(results, desc)


def percent_split(estimators,
                  datasets,
                  split=0.8,
                  nfolds=10,
                  metric=mean_absolute_error,
                  desc=False):
    results = {}
    for dataset_class, datasets_grid in datasets.items():
        for dataset_params in ParameterGrid(datasets_grid):
            for est_class, est_grid in estimators.items():
                for est_params in ParameterGrid(est_grid):
                    estimator = est_class(**est_params)
                    dataset = dataset_class(**dataset_params)
                    train, test = dataset.split(split)
                    fit(estimator, train, desc)
                    scores = score(estimator, test, nfolds, metric)
                    train_repr = re.sub('\s+', ' ', train.__repr__())
                    estimator_repr = re.sub('\s+', ' ', estimator.__repr__())
                    key = (train_repr, estimator_repr)
                    results[key] = scores
    return Results(results, desc)


compare = percent_split


def fit(estimator, dataset, desc):
    logger.info('fitting {} to {}'.format(estimator, dataset))
    if hasattr(estimator, 'partial_fit'):
        sgd(estimator, dataset, desc)
    else:
        bgd(estimator, dataset)


def bgd(estimator, train):
    estimator.fit(train.data, train.target)


def sgd(estimator,
        train,
        desc,
        val_split=0.9,
        max_epochs=1000,
        patience=20,
        batch_size=32):
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


class Results(OrderedDict):
    '''Results are the return type of `Experiment.compare`.

    The results are given as an ordered dict mapping classifiers to the list
    of accuracies for each fold of the cross-validation.

    The `__str__` method is overridden to provide a pretty report of
    classifier accuracies, including a t-test.
    '''

    def __init__(self, results, desc=False):
        super().__init__(
            sorted(results.items(), key=lambda i: np.mean(i[1]), reverse=desc))

        logger.info('performing t-test')
        n = len(results)
        self.ttest = np.zeros((n, n))
        for i, a_scores in enumerate(self.values()):
            for j, b_scores in enumerate(self.values()):
                if i == j:
                    continue
                t, p = sp.stats.ttest_rel(a_scores, b_scores)
                self.ttest[i, j] = p

    def __str__(self):
        str = ''
        str += 'METRIC  TRIAL\n'
        str += '------------------------------------------------------------------------\n'
        for key, scores in self.items():
            str += '{:<7.3f} {}\n'.format(np.mean(scores, key[0]))
            str += ' ' * 8 + '{}\n\n'.format(key[1])
        str += '\n'

        str += 't-Test Matrix (p-values)\n'
        str += '------------------------------------------------------------------------\n'
        for i, row in enumerate(self.ttest):
            for j, p in enumerate(row):
                if i == j:
                    str += '   --    '
                else:
                    str += '{:8.3%} '.format(p)
            str += '\n'
        return str
