#!/usr/bin/env python3
''' Experiment 8: The Kitchen Sink, pt. 2

I took the top 5 algorithms from exp7 and ran them on 6hr windows.

Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
62.299  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+24)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

62.363  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+24)',), lag=4)
        BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

62.657  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+24)',), lag=4)
        ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

67.345  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+24)',), lag=4)
        GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None, learning_rate=0.1, loss='ls', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, presort='auto', random_state=None, subsample=1.0, verbose=0, warm_start=False)

67.345  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+24)',), lag=4)
        XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1, missing=None, n_estimators=100, nthread=-1, objective='reg:linear', reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0, silent=True, subsample=1)


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --     82.412%  28.905%   0.039%   0.039%
 82.412%    --     37.252%   0.047%   0.047%
 28.905%  37.252%    --      0.050%   0.050%
  0.039%   0.047%   0.050%    --     11.238%
  0.039%   0.047%   0.050%  11.238%    --
```
'''

from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, \
	GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import scale as standard_scale

from experiments import core
from data import gaemn15


core.setup()

datasets = [
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'),
        y_features = ('solar radiation (+24)',),
        lag        = 4,
		scale      = standard_scale,
    ),
]

estimators = {
    ExtraTreesRegressor(): {},
    RandomForestRegressor(): {},
    BaggingRegressor(): {},
    GradientBoostingRegressor(): {},
	XGBRegressor(): {},
}

results = core.compare(estimators, datasets, split=0.8, nfolds=10)
print(results)
