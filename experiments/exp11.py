#!/usr/bin/env python3
''' Experiment 11

Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
67.106  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation', 'wind speed', 'wind direction'), y_features=('solar radiation (+96)',), lag=4)
        XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1, missing=None, n_estimators=100, nthread=-1, objective='reg:linear', reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0, silent=True, subsample=1)

67.366  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+96)',), lag=4)
        XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1, missing=None, n_estimators=100, nthread=-1, objective='reg:linear', reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0, silent=True, subsample=1)

67.868  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+24)',), lag=4)
        XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1, missing=None, n_estimators=100, nthread=-1, objective='reg:linear', reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0, silent=True, subsample=1)


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --     73.958%  44.800%
 73.958%    --     69.141%
 44.800%  69.141%    --
```
'''

from xgboost import XGBRegressor

from sklearn.preprocessing import scale as standard_scale

from experiments import core
from data import gaemn15


core.setup()

datasets = {
    gaemn15.DataSet: {
        'path'       : ['./gaemn15.zip'],
        'years'      : [range(2003,2013)],
        'x_features' : [('timestamp (int)', 'timestamp (frac)', 'solar radiation'),
                        ('timestamp (int)', 'timestamp (frac)', 'solar radiation', 'wind speed', 'wind direction'),
                        ('timestamp (int)', 'timestamp (frac)', 'solar radiation', 'air temp', 'humidity', 'rainfall')],
        'y_features' : [('solar radiation (+24)',)],
        'lag'        : [4],
        'scale'      : [standard_scale],
    }],
} # yapf: disable

estimators = {
	XGBRegressor: {},
}

results = core.percent_split(estimators, datasets, 0.8, nfolds=10)
print(results)
