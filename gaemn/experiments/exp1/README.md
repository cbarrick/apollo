# Experiment 1

This experiment evaluates the benefits of using delta features and predicted features. It is broken into 4 parts:

- Part 1 tests whether deltas and perfectly predicted features help 1hr solar predictions.
- Part 2 tests which predicted features are most important.
- Part 3 evaluates how much noise we can tolerate in the predicted features.
- Part 4 repeats part 3 on 6hr predictions.


## Part 1

This experiment evaluates the benefits of using delta features and predicted features to evaluate 1hr solar radiation predictions. We simulate perfect predictions of some features by using the future values.

The regressor used for all experiments is Random Forest.

The results indicate that predicted features increase accuracy (assuming perfect predictions) and delta features on't help much.

The results regarding predicted features may not be terribly valid. By simulating perfect predictions, our conclusions are equivalent to saying that it is easier to predict current solar radiation than to predict future radiation. See part 3 for a more meaningful inquiry on predicted features.

### Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
31.135  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'humidity (+4)', 'rainfall (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

32.301  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'humidity (+4)', 'rainfall (+4)', 'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

35.645  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

36.031  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

t-Test Matrix (p-values)
------------------------------------------------------------------------
   --      0.005%   0.000%   0.000%
  0.005%    --      0.000%   0.000%
  0.000%   0.000%    --      1.750%
  0.000%   0.000%   1.750%    --
```


## Part 2

Part 1 told us that we can predict some features and use those predicted features as additional inputs to predict solar radiation.

This experiment tries to see which of those predicted features are most beneficial. Like part 1, this experiment simulates perfect predictions.

The results seem to indicate that more predicted inputs are better, but predicted rainfall is the most useless. This is equivalent to saying that predicting current solar radiation using current features and some old features is easier than predicting into the future. See part 3 for a more meaningful inquiry on predicted features.

### Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
31.049  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'humidity (+4)', 'rainfall (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

31.394  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'humidity (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

31.733  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'humidity (+4)', 'rainfall (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

32.152  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'humidity (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

32.485  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

32.513  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'rainfall (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

35.195  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'rainfall (+4)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

35.645  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --      6.309%   0.109%   0.002%   0.005%   0.015%   0.000%   0.000%
  6.309%    --      1.517%   0.003%   0.004%   0.003%   0.000%   0.000%
  0.109%   1.517%    --      0.288%   0.085%   0.072%   0.000%   0.000%
  0.002%   0.003%   0.288%    --      3.563%   2.157%   0.000%   0.000%
  0.005%   0.004%   0.085%   3.563%    --     83.538%   0.001%   0.001%
  0.015%   0.003%   0.072%   2.157%  83.538%    --      0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%   0.001%   0.000%    --      3.521%
  0.000%   0.000%   0.000%   0.000%   0.001%   0.000%   3.521%    --
```


## Part 3

Part 2 told us that if we have perfect predictions of certain features, it becomes easier to predict solar radiation. This is equivalent to saying that predicting current solar radiation is easier than predicting future radiation.

This experiment attempts to test that idea more meaningfully by adding noise to the simulated predictions. Each successive trial adds more noise to the predicted features. The noise is drawn from a normal distribution centered around 0 with a standard deviation equal to some multiple of the standard deviation of the prefect prediction.

The results indicate that these predicted features are only useful in predicting solar radiation one hour into the future if the noise between the actual and perfect predictions is less than ~0.5 standard deviations of the perfect signal.

Experiment 5 attempts this experiment on 6hr time windows.

### Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
30.841  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=0)', 'humidity (+4,noise=0)', 'rainfall (+4,noise=0)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

33.460  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=0.1)', 'humidity (+4,noise=0.1)', 'rainfall (+4,noise=0.1)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

34.619  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=0.2)', 'humidity (+4,noise=0.2)', 'rainfall (+4,noise=0.2)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

35.687  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=0.5)', 'humidity (+4,noise=0.5)', 'rainfall (+4,noise=0.5)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

35.688  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

36.220  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=1)', 'humidity (+4,noise=1)', 'rainfall (+4,noise=1)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

36.588  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=2)', 'humidity (+4,noise=2)', 'rainfall (+4,noise=2)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

36.685  DataSet(path='/Users/csb/Desktop/SolarRadiation/gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=3)', 'humidity (+4,noise=3)', 'rainfall (+4,noise=3)'), y_features=('solar radiation (+4)',), lag=4)
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


## Part 4

This experiment repeats part 3 as a six hour prediction problem.

The results are pretty much the same as part 3. Once the noise is greater that 0.5 standard deviations of the perfect prediction, then predicted values do more harm than good.

### Results:
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
