#!/usr/bin/env python3
''' Experiment 1

This experiment evaluates the benefits of using delta features and predicted
features to evaluate 1hr solar radiation predictions. We simulate perfect
predictions of some features by using the future values.

The regressor used for all experiments is Random Forest.

The results indicate that predicted features increase accuracy (assuming perfect
predictions) and delta features on't help much.

The results regarding predicted features may not be terribly valid. By
simulating perfect predictions, our conclusions are equivalent to saying that
it is easier to predict current solar radiation than to predict future
radiation. See experiment 4 for a more meaningful inquiry on predicted
features.

Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
31.135	DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'humidity (+4)', 'rainfall (+4)'), y_features=('solar radiation (+4)',), lag=4)
		RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

32.301	DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'humidity (+4)', 'rainfall (+4)', 'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'), y_features=('solar radiation (+4)',), lag=4)
		RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

35.645	DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
		RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

36.031	DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'), y_features=('solar radiation (+4)',), lag=4)
		RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

t-Test Matrix (p-values)
------------------------------------------------------------------------
   --      0.005%   0.000%   0.000%
  0.005%    --      0.000%   0.000%
  0.000%   0.000%    --      1.750%
  0.000%   0.000%   1.750%    --
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
                      'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4)', 'humidity (+4)', 'rainfall (+4)',),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4)', 'humidity (+4)', 'rainfall (+4)',
                      'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
]

estimators = {
    RandomForestRegressor(): {},
}

results = core.compare(estimators, datasets, split=0.8, nfolds=10)
print(results)
