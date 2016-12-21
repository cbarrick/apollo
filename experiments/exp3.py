#!/usr/bin/env python3
''' Experiment 3

Experiment 1 told us that we can predict some features and use those predicted
features as additional inputs to predict solar radiation.

This experiment tries to see which of those predicted features are most
beneficial. Like experiment 1, this experiment simulates perfect predictions.

The results seem to indicate that more predicted inputs are better, but
predicted rainfall is the most useless. This is equivalent to saying that
predicting current solar radiation using current features and some old features
is easier than predicting into the future. See experiment 4 for a more
meaningful inquiry on predicted features.

Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
31.049  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'humidity (+4)', 'rainfall (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

31.394  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'humidity (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

31.733  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'humidity (+4)', 'rainfall (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

32.152  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'humidity (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

32.485  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

32.513  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'rainfall (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

35.195  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'rainfall (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

35.645  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --      6.309%   0.109%   0.002%   0.005%   0.015%   0.000%   0.000%
  6.309%    --      1.517%   0.003%   0.004%   0.003%   0.000%   0.000%
  0.109%   1.517%    --      0.288%   0.085%   0.072%   0.000%   0.000%
  0.002%   0.003%   0.288%    --      3.563%   2.157%   0.000%   0.000%
  0.005%   0.004%   0.085%   3.563%    --     83.538%   0.001%   0.001%
  0.015%   0.003%   0.072%   2.157%  83.538%    --      0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%   0.001%   0.000%    --      3.521%
  0.000%   0.000%   0.000%   0.000%   0.001%   0.000%   3.521%    --
```
'''

from sklearn.ensemble import RandomForestRegressor

from experiments import core
from data import gaemn15

core.setup()

datasets = [
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4)',),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'humidity (+4)',),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'rainfall (+4)',),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4)', 'humidity (+4)'),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'humidity (+4)', 'rainfall (+4)'),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4)', 'rainfall (+4)'),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4)', 'humidity (+4)', 'rainfall (+4)'),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
]

estimators = {
    RandomForestRegressor(): {},
}

results = core.compare(estimators, datasets, split=0.8)
print(results)
