import logging
import math
import re
from collections import OrderedDict

import numpy as np
import scipy as sp

from scipy import stats
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error

from data import gaemn15


def setup():
    '''Parses CLI arguments and sets global configuration.'''
    import argparse

    parser = argparse.ArgumentParser(description='Perform the experiment')
    parser.add_argument('--log', default='WARNING', type=str, help='the level of logging details')
    parser.add_argument('--seed', default=1337, type=int, help='the random seed of the experiment')
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log)
    logging.basicConfig(level=numeric_level)

    np.random.seed(args.seed)


def compare(estimators,
        datasets=[gaemn15.DataSet()],
        split=0.8,
        nfolds=10,
        metric=mean_absolute_error,
        desc=False):
    '''Compare estimators against the datasets.

    The results are returned as a `Results` object.
    '''
    results = {}
    space = re.compile('\s+')
    for estimator, grid in estimators.items():
        for params in ParameterGrid(grid):
            for dataset in datasets:
                estimator.set_params(**params)
                dataset_repr = space.sub(' ', dataset.__repr__())
                estimator_repr = space.sub(' ', estimator.__repr__())
                key = (dataset_repr, estimator_repr)
                logging.info('fitting {1} to {0}'.format(*key))
                s = int(len(dataset.data) * split)
                estimator.fit(dataset.data[:s], dataset.target[:s])
                n = len(dataset.data[s:]) // nfolds
                for i in range(nfolds):
                    test_slice = slice(s+n*i,s+n*(i+1))
                    pred = estimator.predict(dataset.data[test_slice])
                    score = metric(dataset.target[test_slice], pred)
                    try:
                        results[key].append(score)
                    except:
                        results[key] = [score]
    return Results(results, desc)


class Results(OrderedDict):
    '''Results are the return type of `Experiment.compare`.

    The results are given as an ordered dict mapping classifiers to the list
    of accuracies for each fold of the cross-validation.

    The `__str__` method is overridden to provide a pretty report of
    classifier accuracies, including a t-test.
    '''
    def __init__(self, results, desc=False):
        super().__init__(sorted(results.items(), key=lambda i:np.mean(i[1]), reverse=desc))

        logging.info('performing t-test')
        n = len(results)
        self.ttest = np.zeros((n,n))
        for i, a_scores in enumerate(self.values()):
            for j, b_scores in enumerate(self.values()):
                if i == j:
                    continue
                t, p = sp.stats.ttest_rel(a_scores, b_scores)
                self.ttest[i,j] = p

    def __str__(self):
        str = ''
        str += 'METRIC  TRIAL\n'
        str += '------------------------------------------------------------------------\n'
        for key, scores in self.items():
            str += '{:.3f}\t{}\n\t\t{}\n\n'.format(np.mean(scores), key[0], key[1])
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
