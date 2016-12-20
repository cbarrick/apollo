#!/usr/bin/env python3
''' Experiment 8: The Kitchen Sink, pt. 2

I took the top 5 algorithms from exp7 and ran them on 6hr windows.

[Extra Trees][1] are winning.

[1] P. Geurts, D. Ernst., and L. Wehenkel, “Extremely randomized trees”,
Machine Learning, 63(1), 3-42, 2006.

Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
40.552	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+24)',), lag=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('extratreesregressor', ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))])

44.339	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+24)',), lag=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('baggingregressor', BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))])

44.425	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+24)',), lag=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('randomforestregressor', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))])

65.749	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+24)',), lag=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('gradientboostingregressor', GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None, learning_rate=0.1, loss='ls', max_depth=3, max_features=None, max_leaf_nodes=None, mi... presort='auto', random_state=None, subsample=1.0, verbose=0, warm_start=False))])


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --      0.000%   0.000%   0.000%
  0.000%    --     63.158%   0.000%
  0.000%  63.158%    --      0.000%
  0.000%   0.000%   0.000%    --
```
'''

from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, \
	ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, \
	HuberRegressor, Lars, Lasso, LinearRegression, OrthogonalMatchingPursuit, \
	PassiveAggressiveRegressor, Perceptron, RANSACRegressor, Ridge, \
	TheilSenRegressor

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    ),
]

estimators = {
	make_pipeline(StandardScaler(), ExtraTreesRegressor()): {},
	make_pipeline(StandardScaler(), RandomForestRegressor()): {},
	make_pipeline(StandardScaler(), BaggingRegressor()): {},
	make_pipeline(StandardScaler(), GradientBoostingRegressor()): {},
}

results = core.compare(estimators, datasets, split=0.8, nfolds=10)
print(results)
