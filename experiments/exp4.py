#!/usr/bin/env python3
''' Experiment 4

Experiment 3 told us that if we have perfect predictions of certain features,
it becomes easier to predict solar radiation. This is equivalent to saying that
predicting current solar radiation is easier than predicting future radiation.

This experiment attempts to test that idea more meaningfully by adding noise to
the simulated predictions. Each successive trial adds more noise to the
predicted features. The noise is drawn from a normal distribution centered
around 0 with a standard deviation equal to some multiple of the standard
deviation of the prefect prediction.

The results indicate that these predicted features are only useful in predicting
solar radiation one hour into the future if the noise between the actual and
perfect predictions is less than ~0.5 standard deviations of the perfect signal.

Experiment 5 attempts this experiment on 6hr time windows.

Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
30.841	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=0)', 'humidity (+4,noise=0)', 'rainfall (+4,noise=0)'), y_features=('solar radiation (+4)',), lag=4)
		RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

33.460	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=0.1)', 'humidity (+4,noise=0.1)', 'rainfall (+4,noise=0.1)'), y_features=('solar radiation (+4)',), lag=4)
		RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

34.619	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=0.2)', 'humidity (+4,noise=0.2)', 'rainfall (+4,noise=0.2)'), y_features=('solar radiation (+4)',), lag=4)
		RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

35.687	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=0.5)', 'humidity (+4,noise=0.5)', 'rainfall (+4,noise=0.5)'), y_features=('solar radiation (+4)',), lag=4)
		RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

35.688	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
		RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

36.220	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=1)', 'humidity (+4,noise=1)', 'rainfall (+4,noise=1)'), y_features=('solar radiation (+4)',), lag=4)
		RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

36.588	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=2)', 'humidity (+4,noise=2)', 'rainfall (+4,noise=2)'), y_features=('solar radiation (+4)',), lag=4)
		RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

36.685	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=3)', 'humidity (+4,noise=3)', 'rainfall (+4,noise=3)'), y_features=('solar radiation (+4)',), lag=4)
		RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --      0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%
  0.000%    --      0.002%   0.000%   0.000%   0.000%   0.000%   0.000%
  0.000%   0.002%    --      0.001%   0.002%   0.000%   0.000%   0.000%
  0.000%   0.000%   0.001%    --     99.256%   0.355%   0.069%   0.005%
  0.000%   0.000%   0.002%  99.256%    --      0.067%   0.015%   0.000%
  0.000%   0.000%   0.000%   0.355%   0.067%    --      0.896%   0.591%
  0.000%   0.000%   0.000%   0.069%   0.015%   0.896%    --     45.717%
  0.000%   0.000%   0.000%   0.005%   0.000%   0.591%  45.717%    --
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
                      'air temp (+4,noise=0)', 'humidity (+4,noise=0)', 'rainfall (+4,noise=0)'),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4,noise=0.1)', 'humidity (+4,noise=0.1)', 'rainfall (+4,noise=0.1)'),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4,noise=0.2)', 'humidity (+4,noise=0.2)', 'rainfall (+4,noise=0.2)'),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4,noise=0.5)', 'humidity (+4,noise=0.5)', 'rainfall (+4,noise=0.5)'),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4,noise=1)', 'humidity (+4,noise=1)', 'rainfall (+4,noise=1)'),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4,noise=2)', 'humidity (+4,noise=2)', 'rainfall (+4,noise=2)'),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4,noise=3)', 'humidity (+4,noise=3)', 'rainfall (+4,noise=3)'),
        y_features = ('solar radiation (+4)',),
        lag        = 4,
    ),
]

estimators = {
    RandomForestRegressor(): {},
}

results = core.compare(estimators, datasets, split=0.8)
print(results)
