#!/usr/bin/env python3
''' Experiment 6

This experiment compares different scaling schemes under a random forest
regrssion. The result is that no kind of scaling has a significant effect.

The results are likely different for neural nets.

Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
35.653	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
		Pipeline(steps=[('predict', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))])

35.686	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
		Pipeline(steps=[('min_max_scale', MinMaxScaler(copy=True, feature_range=(0, 1))), ('predict', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))])

35.730	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
		Pipeline(steps=[('stdandardize', StandardScaler(copy=True, with_mean=True, with_std=True)), ('predict', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))])

35.954	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
		Pipeline(steps=[('max_abs_scale', MaxAbsScaler(copy=True)), ('predict', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))])


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --     78.408%  49.600%  10.503%
 78.408%    --     67.331%  10.099%
 49.600%  67.331%    --     23.230%
 10.503%  10.099%  23.230%    --
```
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

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
]

estimators = {
    Pipeline([
		('predict', RandomForestRegressor()),
	]): {},

	Pipeline([
		('stdandardize', StandardScaler()),
		('predict', RandomForestRegressor()),
	]): {},

	Pipeline([
		('min_max_scale', MinMaxScaler()),
		('predict', RandomForestRegressor()),
	]): {},

	Pipeline([
		('max_abs_scale', MaxAbsScaler()),
		('predict', RandomForestRegressor()),
	]): {},
}

results = core.compare(estimators, datasets, split=0.8, nfolds=10)
print(results)
