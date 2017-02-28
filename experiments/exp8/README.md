# Experiment 8

This experiment strives to find the best model possible for 1hr predictions.


## Part 1

This first part tries to find the optimal lag for 1hr predictions.

The results indicate that lags of 12 (3hr) and 24 (6hrs) are significantly better than no lag. A lag of 96 (24hr) is no better than a lag of 4 (1hr).

### Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
37.548  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

37.713  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=12)

38.112  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)

38.563  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=96)


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --     43.486%   4.236%  10.852%
 43.486%    --      2.704%   9.503%
  4.236%   2.704%    --     37.217%
 10.852%   9.503%  37.217%    --    
```


## Part 2

This part tests feature choice for 1hr predictions using the 6hr lag discovered from part 1.

The results are not statistically significant, but show that using only solar radiation is probably the best choice.

### Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
37.819  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

38.097  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation', 'wind speed', 'wind direction', 'barometric pressure'), y_features=('solar radiation (+4)',), lag=24)

38.170  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation', 'wind speed', 'wind direction'), y_features=('solar radiation (+4)',), lag=24)


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --     50.033%  32.721%
 50.033%    --     67.529%
 32.721%  67.529%    --  
 ```


## Part 3

This part does hyper parameter tuning for the best models found in experiment 4 (excluding XGBoost due to some compilation issues). The results indicate that Extremely Randomized Trees, Random Forrest, and Bagging are all pretty much the same, and that more trees is better.

I am using only timestamps and solar radiation (per part 2) and a 6hr lag (per part 1).

### Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
36.125  ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

36.154  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

36.155  BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_estimators=500, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

36.171  ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

36.178  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

36.220  ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

36.222  BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_estimators=200, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

36.266  BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_estimators=100, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

36.287  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

36.350  ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

36.366  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

36.463  BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_estimators=50, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

36.674  BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_estimators=25, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

36.720  ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=25, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

36.734  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=25, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

37.461  ExtraTreesRegressor(bootstrap=False, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

37.706  BaggingRegressor(base_estimator=None, bootstrap=True, bootstrap_features=False, max_features=1.0, max_samples=1.0, n_estimators=10, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

37.819  RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1, oob_score=False, random_state=None, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

37.831  GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None, learning_rate=0.1, loss='ls', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=500, presort='auto', random_state=None, subsample=1.0, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

38.566  GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None, learning_rate=0.1, loss='ls', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=200, presort='auto', random_state=None, subsample=1.0, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

39.571  GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None, learning_rate=0.1, loss='ls', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=100, presort='auto', random_state=None, subsample=1.0, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

42.141  GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None, learning_rate=0.1, loss='ls', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, presort='auto', random_state=None, subsample=1.0, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

55.535  GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None, learning_rate=0.1, loss='ls', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=25, presort='auto', random_state=None, subsample=1.0, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)

107.256 GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None, learning_rate=0.1, loss='ls', max_depth=3, max_features=None, max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, presort='auto', random_state=None, subsample=1.0, verbose=0, warm_start=False)
        DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'solar radiation'), y_features=('solar radiation (+4)',), lag=24)


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --     88.359%  87.886%  13.850%  78.972%   2.662%  66.914%  49.362%  41.654%   0.011%  28.102%   8.154%   0.856%   0.000%   2.771%   0.003%   0.037%   0.015%   0.060%   0.024%   0.011%   0.001%   0.000%   0.000%
 88.359%    --     92.693%  92.851%  31.679%  72.085%  10.338%   0.388%   0.005%  31.688%   0.198%   0.036%   0.002%   1.796%   0.005%   0.044%   0.000%   0.000%   0.052%   0.014%   0.004%   0.000%   0.000%   0.000%
 87.886%  92.693%    --     93.498%  43.941%  72.797%   8.568%   0.601%   0.015%  32.620%   0.455%   0.029%   0.004%   1.913%   0.002%   0.041%   0.000%   0.000%   0.059%   0.016%   0.004%   0.000%   0.000%   0.000%
 13.850%  92.851%  93.498%    --     96.995%  15.207%  81.725%  63.518%  54.305%   0.037%  35.555%  11.592%   1.014%   0.000%   3.781%   0.003%   0.033%   0.012%   0.091%   0.032%   0.013%   0.001%   0.000%   0.000%
 78.972%  31.679%  43.941%  96.995%    --     82.329%  23.055%   0.324%   0.876%  38.527%   0.771%   0.116%   0.011%   2.196%   0.003%   0.051%   0.000%   0.000%   0.042%   0.012%   0.003%   0.000%   0.000%   0.000%
  2.662%  72.085%  72.797%  15.207%  82.329%    --     99.295%  81.248%  71.799%   3.014%  48.114%  16.642%   1.582%   0.001%   4.899%   0.002%   0.038%   0.013%   0.095%   0.033%   0.013%   0.001%   0.000%   0.000%
 66.914%  10.338%   8.568%  81.725%  23.055%  99.295%    --     27.681%  14.399%  56.430%   5.277%   0.997%   0.059%   5.085%   0.001%   0.089%   0.000%   0.000%   0.078%   0.017%   0.004%   0.000%   0.000%   0.000%
 49.362%   0.388%   0.601%  63.518%   0.324%  81.248%  27.681%    --     48.481%  67.155%  11.874%   0.376%   0.082%   4.990%   0.007%   0.088%   0.000%   0.000%   0.072%   0.018%   0.005%   0.000%   0.000%   0.000%
 41.654%   0.005%   0.015%  54.305%   0.876%  71.799%  14.399%  48.481%    --     73.864%  18.621%   0.375%   0.013%   5.176%   0.026%   0.075%   0.000%   0.000%   0.109%   0.024%   0.006%   0.000%   0.000%   0.000%
  0.011%  31.688%  32.620%   0.037%  38.527%   3.014%  56.430%  67.155%  73.864%    --     93.875%  51.423%   6.088%   0.002%  12.307%   0.009%   0.069%   0.034%   0.159%   0.044%   0.016%   0.002%   0.000%   0.000%
 28.102%   0.198%   0.455%  35.555%   0.771%  48.114%   5.277%  11.874%  18.621%  93.875%    --     31.245%   0.521%  12.981%   0.433%   0.233%   0.000%   0.000%   0.116%   0.024%   0.006%   0.000%   0.000%   0.000%
  8.154%   0.036%   0.029%  11.592%   0.116%  16.642%   0.997%   0.376%   0.375%  51.423%  31.245%    --      2.744%  17.383%   2.471%   0.106%   0.003%   0.001%   0.299%   0.059%   0.013%   0.001%   0.000%   0.000%
  0.856%   0.002%   0.004%   1.014%   0.011%   1.582%   0.059%   0.082%   0.013%   6.088%   0.521%   2.744%    --     78.419%  61.533%   0.541%   0.010%   0.004%   0.565%   0.073%   0.013%   0.001%   0.000%   0.000%
  0.000%   1.796%   1.913%   0.000%   2.196%   0.001%   5.085%   4.990%   5.176%   0.002%  12.981%  17.383%  78.419%    --     95.501%   0.044%   0.517%   0.246%   0.896%   0.153%   0.039%   0.003%   0.000%   0.000%
  2.771%   0.005%   0.002%   3.781%   0.003%   4.899%   0.001%   0.007%   0.026%  12.307%   0.433%   2.471%  61.533%  95.501%    --      2.372%   0.011%   0.022%   0.704%   0.067%   0.009%   0.000%   0.000%   0.000%
  0.003%   0.044%   0.041%   0.003%   0.051%   0.002%   0.089%   0.088%   0.075%   0.009%   0.233%   0.106%   0.541%   0.044%   2.372%    --     39.048%  20.159%  42.093%   5.451%   0.565%   0.011%   0.000%   0.000%
  0.037%   0.000%   0.000%   0.033%   0.000%   0.038%   0.000%   0.000%   0.000%   0.069%   0.000%   0.003%   0.010%   0.517%   0.011%  39.048%    --     45.568%  74.639%   6.576%   0.289%   0.003%   0.000%   0.000%
  0.015%   0.000%   0.000%   0.012%   0.000%   0.013%   0.000%   0.000%   0.000%   0.034%   0.000%   0.001%   0.004%   0.246%   0.022%  20.159%  45.568%    --     97.715%  15.746%   1.035%   0.009%   0.000%   0.000%
  0.060%   0.052%   0.059%   0.091%   0.042%   0.095%   0.078%   0.072%   0.109%   0.159%   0.116%   0.299%   0.565%   0.896%   0.704%  42.093%  74.639%  97.715%    --      0.048%   0.018%   0.001%   0.000%   0.000%
  0.024%   0.014%   0.016%   0.032%   0.012%   0.033%   0.017%   0.018%   0.024%   0.044%   0.024%   0.059%   0.073%   0.153%   0.067%   5.451%   6.576%  15.746%   0.048%    --      0.014%   0.001%   0.000%   0.000%
  0.011%   0.004%   0.004%   0.013%   0.003%   0.013%   0.004%   0.005%   0.006%   0.016%   0.006%   0.013%   0.013%   0.039%   0.009%   0.565%   0.289%   1.035%   0.018%   0.014%    --      0.000%   0.000%   0.000%
  0.001%   0.000%   0.000%   0.001%   0.000%   0.001%   0.000%   0.000%   0.000%   0.002%   0.000%   0.001%   0.001%   0.003%   0.000%   0.011%   0.003%   0.009%   0.001%   0.001%   0.000%    --      0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%    --      0.000%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%    --    
  ```
