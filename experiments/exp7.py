#!/usr/bin/env python3
''' Experiment 7: The Kitchen Sink

In this experiment, we throw most of the regression algorithms supplied by
Scikit at our basic dataset. We don't include neural nets since they take so
long train, and we don't include some of the hybrid linear models because we
only want to compare the basic components for now.

Some of the regressors were crashing on me. Probably due to out-of-memory
errors. Those have been commented out.

The results indicate that Extra Trees is an improvement over Random Forest that
I have been using. Bagging also did alright, but worse than Random Forest.

Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
35.141	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('extratreesregressor', ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))])

35.693	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('randomforestregressor', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))])

35.698	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('baggingregressor', BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))])

39.133	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('gradientboostingregressor', GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None, learning_rate=0.1, loss='ls', max_depth=3, max_features=None, max_leaf_nodes=None, mi... presort='auto', random_state=None, subsample=1.0, verbose=0, warm_start=False))])

62.854	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('adaboostregressor', AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear', n_estimators=50, random_state=None))])

64.376	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('huberregressor', HuberRegressor(alpha=0.0001, epsilon=1.35, fit_intercept=True, max_iter=100, tol=1e-05, warm_start=False))])

70.944	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('passiveaggressiveregressor', PassiveAggressiveRegressor(C=1.0, epsilon=0.1, fit_intercept=True, loss='epsilon_insensitive', n_iter=5, random_state=None, shuffle=True, verbose=0, warm_start=False))])

73.044	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('theilsenregressor', TheilSenRegressor(copy_X=True, fit_intercept=True, max_iter=300, max_subpopulation=10000, n_jobs=1, n_subsamples=None, random_state=None, tol=0.001, verbose=False))])

73.820	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('linearregression', LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False))])

73.820	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('bayesianridge', BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300, normalize=False, tol=0.001, verbose=False))])

74.707	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('lasso', Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000, normalize=False, positive=False, precompute=False, random_state=None, selection='cyclic', tol=0.0001, warm_start=False))])

76.990	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('orthogonalmatchingpursuit', OrthogonalMatchingPursuit(fit_intercept=True, n_nonzero_coefs=None, normalize=True, precompute='auto', tol=None))])

91.441	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('lars', Lars(copy_X=True, eps=2.2204460492503131e-16, fit_intercept=True, fit_path=True, n_nonzero_coefs=500, normalize=True, positive=False, precompute='auto', verbose=False))])

92.560	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('elasticnet', ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5, max_iter=1000, normalize=False, positive=False, precompute=False, random_state=None, selection='cyclic', tol=0.0001, warm_start=False))])

149.519	DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), window=4)
		Pipeline(steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('ransacregressor', RANSACRegressor(base_estimator=None, is_data_valid=None, is_model_valid=None, loss='absolute_loss', max_trials=100, min_samples=None, random_state=None, residual_metric=None, residual_threshold=None, stop_n_inliers=inf, stop_probability=0.99, stop_score=inf))])


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --      0.103%   0.172%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%
  0.103%    --     95.926%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%
  0.172%  95.926%    --      0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%
  0.000%   0.000%   0.000%    --      0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%    --     12.417%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%  12.417%    --      0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%    --      0.168%   0.005%   0.005%   0.003%   0.000%   0.000%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.168%    --     36.538%  36.529%   9.508%   0.007%   0.002%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.005%  36.538%    --     68.662%   0.002%   0.001%   0.001%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.005%  36.529%  68.662%    --      0.002%   0.001%   0.001%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.003%   9.508%   0.002%   0.002%    --      0.081%   0.001%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.007%   0.001%   0.001%   0.081%    --      0.005%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.002%   0.001%   0.001%   0.001%   0.005%    --     57.983%   0.001%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%  57.983%    --      0.000%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.001%   0.000%    --

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
        y_features = ('solar radiation (+4)',),
        window     = 4,
    ),
]

estimators = {
	make_pipeline(StandardScaler(), AdaBoostRegressor()): {},
	make_pipeline(StandardScaler(), BaggingRegressor()): {},
	make_pipeline(StandardScaler(), ExtraTreesRegressor()): {},
	make_pipeline(StandardScaler(), GradientBoostingRegressor()): {},
	make_pipeline(StandardScaler(), RandomForestRegressor()): {},
	# make_pipeline(StandardScaler(), GaussianProcessRegressor()): {},
	# make_pipeline(StandardScaler(), KernelRidge()): {},
	# make_pipeline(StandardScaler(), ARDRegression()): {},
	make_pipeline(StandardScaler(), BayesianRidge()): {},
	make_pipeline(StandardScaler(), ElasticNet()): {},
	make_pipeline(StandardScaler(), HuberRegressor()): {},
	make_pipeline(StandardScaler(), Lars()): {},
	make_pipeline(StandardScaler(), Lasso()): {},
	make_pipeline(StandardScaler(), LinearRegression()): {},
	make_pipeline(StandardScaler(), OrthogonalMatchingPursuit()): {},
	make_pipeline(StandardScaler(), PassiveAggressiveRegressor()): {},
	# make_pipeline(StandardScaler(), Perceptron()): {},
	make_pipeline(StandardScaler(), RANSACRegressor()): {},
	# make_pipeline(StandardScaler(), Ridge()): {},
	make_pipeline(StandardScaler(), TheilSenRegressor()): {},
}

results = core.compare(estimators, datasets, split=0.8, nfolds=10)
print(results)
