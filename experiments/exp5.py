#!/usr/bin/env python3
''' Experiment 5

This experiment repeates experiment 4 as a six hour prediction problem.

The results are pretty much the same as experiment 4. Once the noise is greater
that 0.5 standard deviations of the perfect prediction, then predicted values
do more harm than good.

Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
40.508  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=0)', 'humidity (+24,noise=0)', 'rainfall (+24,noise=0)'), y_features=('solar radiation (+24)',), lag=24)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

43.649  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=0.1)', 'humidity (+24,noise=0.1)', 'rainfall (+24,noise=0.1)'), y_features=('solar radiation (+24)',), lag=24)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

45.815  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=0.2)', 'humidity (+24,noise=0.2)', 'rainfall (+24,noise=0.2)'), y_features=('solar radiation (+24)',), lag=24)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

49.699  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=0.5)', 'humidity (+24,noise=0.5)', 'rainfall (+24,noise=0.5)'), y_features=('solar radiation (+24)',), lag=24)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

56.038  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+24)',), lag=24)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

56.303  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=1)', 'humidity (+24,noise=1)', 'rainfall (+24,noise=1)'), y_features=('solar radiation (+24)',), lag=24)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

59.719  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=3)', 'humidity (+24,noise=3)', 'rainfall (+24,noise=3)'), y_features=('solar radiation (+24)',), lag=24)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

60.029  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=2)', 'humidity (+24,noise=2)', 'rainfall (+24,noise=2)'), y_features=('solar radiation (+24)',), lag=24)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --      0.002%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%
  0.002%    --      0.006%   0.000%   0.000%   0.000%   0.000%   0.000%
  0.000%   0.006%    --      0.000%   0.000%   0.000%   0.000%   0.000%
  0.000%   0.000%   0.000%    --      0.019%   0.000%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.019%    --     82.035%   0.992%   0.261%
  0.000%   0.000%   0.000%   0.000%  82.035%    --      0.002%   0.000%
  0.000%   0.000%   0.000%   0.000%   0.992%   0.002%    --     34.202%
  0.000%   0.000%   0.000%   0.000%   0.261%   0.000%  34.202%    --
```
'''

from sklearn.ensemble import RandomForestRegressor

from experiments import core
from data import gaemn15

core.setup()

datasets = {
    gaemn15.DataSet: [{
        'path'       : ['./gaemn15.zip'],
        'years'      : [range(2003,2013)],
        'x_features' : [('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation')],
                        ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=0)', 'humidity (+24,noise=0)', 'rainfall (+24,noise=0)'),
                        ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=0.1)', 'humidity (+24,noise=0.1)', 'rainfall (+24,noise=0.1)'),
                        ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=0.2)', 'humidity (+24,noise=0.2)', 'rainfall (+24,noise=0.2)'),
                        ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=0.5)', 'humidity (+24,noise=0.5)', 'rainfall (+24,noise=0.5)'),
                        ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=1)', 'humidity (+24,noise=1)', 'rainfall (+24,noise=1)'),
                        ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=2)', 'humidity (+24,noise=2)', 'rainfall (+24,noise=2)'),
                        ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=3)', 'humidity (+24,noise=3)', 'rainfall (+24,noise=3)')],
        'y_features' : [('solar radiation (+24)',)],
        'lag'        : [24],
    }],
} # yapf: disable


estimators = {
    RandomForestRegressor: {},
}

results = core.compare(estimators, datasets, split=0.8)
print(results)
