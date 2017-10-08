# Experiment 4

This experiment attempts to find the best models out of those supplied by Scikit-learn and XGBoost.

- Part 1 runs all of the algorithms on a 1hr prediction test.
- Part 2 takes the top 5 and runs them on a 6hr prediction test.
- Part 3 takes the top 5 and runs them on a 24hr prediction test.


## Part 1

In this experiment, I throw all of the candidate algorithms supplied at a 1hr prediction test.

The results indicate that Extra Trees is an improvement over Random Forest that I have been using. Bagging also did alright, but worse than Random Forest.

### Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
37.641  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

37.658  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

37.956  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

39.901  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1, missing=None, n_estimators=100, nthread=-1, objective='reg:linear', reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0, silent=True, subsample=1)

39.955  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None, learning_rate=0.1, loss='ls', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, presort='auto', random_state=None, subsample=1.0, verbose=0, warm_start=False)

65.821  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        HuberRegressor(alpha=0.0001, epsilon=1.35, fit_intercept=True, max_iter=100, tol=1e-05, warm_start=False)

66.798  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        AdaBoostRegressor(base_estimator=None, learning_rate=1.0, loss='linear', n_estimators=50, random_state=None)

73.894  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        TheilSenRegressor(copy_X=True, fit_intercept=True, max_iter=300, max_subpopulation=10000, n_jobs=1, n_subsamples=None, random_state=None, tol=0.001, verbose=False)

75.676  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

75.677  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True, fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300, normalize=False, tol=0.001, verbose=False)

75.745  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        Lars(copy_X=True, eps=2.2204460492503131e-16, fit_intercept=True, fit_path=True, n_nonzero_coefs=500, normalize=True, positive=False, precompute='auto', verbose=False)

76.849  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        Lasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000, normalize=False, positive=False, precompute=False, random_state=None, selection='cyclic', tol=0.0001, warm_start=False)

77.711  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        OrthogonalMatchingPursuit(fit_intercept=True, n_nonzero_coefs=None, normalize=True, precompute='auto', tol=None)

86.758  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        PassiveAggressiveRegressor(C=1.0, epsilon=0.1, fit_intercept=True, loss='epsilon_insensitive', n_iter=5, random_state=None, shuffle=True, verbose=0, warm_start=False)

95.190  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        ElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5, max_iter=1000, normalize=False, positive=False, precompute=False, random_state=None, selection='cyclic', tol=0.0001, warm_start=False)

6532524053.514 DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        RANSACRegressor(base_estimator=None, is_data_valid=None, is_model_valid=None, loss='absolute_loss', max_trials=100, min_samples=None, random_state=None, residual_metric=None, residual_threshold=None, stop_n_inliers=inf, stop_probability=0.99, stop_score=inf)


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --     94.046%  32.340%   0.076%   0.060%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.001%   0.000%   1.469%
 94.046%    --     14.165%   0.473%   0.377%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.001%   0.000%   1.469%
 32.340%  14.165%    --      1.434%   1.149%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.001%   0.000%   1.469%
  0.076%   0.473%   1.434%    --     12.943%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.001%   0.000%   1.469%
  0.060%   0.377%   1.149%  12.943%    --      0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.001%   0.000%   1.469%
  0.000%   0.000%   0.000%   0.000%   0.000%    --     64.556%   0.000%   0.008%   0.008%   0.008%   0.008%   0.001%   0.677%   0.000%   1.469%
  0.000%   0.000%   0.000%   0.000%   0.000%  64.556%    --      1.529%   0.038%   0.038%   0.036%   0.015%   0.012%   0.710%   0.000%   1.469%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   1.529%    --     30.874%  30.867%  29.366%  14.495%   3.090%   6.542%   0.000%   1.469%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.008%   0.038%  30.874%    --     30.147%   0.583%   0.058%   0.356%   7.668%   0.000%   1.469%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.008%   0.038%  30.867%  30.147%    --      0.641%   0.057%   0.357%   7.669%   0.000%   1.469%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.008%   0.036%  29.366%   0.583%   0.641%    --      0.089%   0.454%   7.816%   0.000%   1.469%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.008%   0.015%  14.495%   0.058%   0.057%   0.089%    --     21.334%  10.235%   0.000%   1.469%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.001%   0.012%   3.090%   0.356%   0.357%   0.454%  21.334%    --     13.780%   0.000%   1.469%
  0.001%   0.001%   0.001%   0.001%   0.001%   0.677%   0.710%   6.542%   7.668%   7.669%   7.816%  10.235%  13.780%    --     18.143%   1.469%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%  18.143%    --      1.469%
  1.469%   1.469%   1.469%   1.469%   1.469%   1.469%   1.469%   1.469%   1.469%   1.469%   1.469%   1.469%   1.469%   1.469%   1.469%    --
```


## Part 2

I took the top 5 algorithms from part 1 and ran them on 6hr windows.

### Results:
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


## Part 3

Like part 2, but with 1 day predictions and only using solar radiation as an input feature.

### Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
67.258  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+96)',), lag=4)
        GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None, learning_rate=0.1, loss='ls', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, presort='auto', random_state=None, subsample=1.0, verbose=0, warm_start=False)

67.366  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+96)',), lag=4)
        XGBRegressor(base_score=0.5, colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1, missing=None, n_estimators=100, nthread=-1, objective='reg:linear', reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0, silent=True, subsample=1)

68.305  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+96)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

68.826  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+96)',), lag=4)
        BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

69.226  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+96)',), lag=4)
        ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --     12.242%  22.747%   8.453%   3.546%
 12.242%    --     26.692%   9.811%   3.945%
 22.747%  26.692%    --      0.486%   2.849%
  8.453%   9.811%   0.486%    --     20.409%
  3.546%   3.945%   2.849%  20.409%    --
```
