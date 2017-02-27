# Experiment 2

This experiment is designed to test the various timestamp features of the dataset. We have the following features:

- `day` gives the day of the year, between 1 and 366.
- `time` gives the time of day like 145 for 1:45AM and 200 for 2:00AM.
- `timestamp` gives a floating point timestamp. The integer part is the day of
  the year and the fractional part is the time of day.

The problem with `time` is that the value increases by 15 within the hour, e.g. from 100 (1:00AM) to 115 (1:15AM), and by 55 between hours, e.g. 145 (1:45AM) to 200 (2:00AM). The problem with `timestamp` is that it combines two periodic signals, the day of year and time of day. Ideally, we'd like separate uniform signals for day of year and time of day.

This experiment uses Random Forest to compare the use of the features `day` and `time` together vs. `timestamp` alone vs. `day` and `timestamp` together vs. a realization of the ideal signals.

The results show that the ideal signal is almost significatly better than using `day` and `time` together, which is better than using `timestamp` alone or `day` and `timestamp` together.

This experiment applies to Random Forest. My gut tells me that neural nets may prefer the ideal signal more heavily.

### Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
35.527  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp (int)', 'timestamp (frac)', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

35.976  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

41.786  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'timestamp', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

41.922  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('timestamp', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'), y_features=('solar radiation (+4)',), lag=4)
        RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False)

t-Test Matrix (p-values)
------------------------------------------------------------------------
   --      5.110%   0.000%   0.000%
  5.110%    --      0.000%   0.000%
  0.000%   0.000%    --     36.613%
  0.000%   0.000%  36.613%    --
```
