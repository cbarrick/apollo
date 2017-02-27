# Experiment 5

This experiment looks at feature selection. The most important result is that solar radiation is the most useful feature to predict future solar radiation. Wind features may also be helpful.


## Part 1

This experiment trains a random forrest regressor on a 24 hour prediction with no lag. The model's importance ranking of each feature is then presented.

Interpreting this output is tricky. PAR is ranked as the most important by a large amount, but because it is highly correlated with Solar Radiation, I believe this is telling us that current solar radiation is by far the most important feature. The timestamp features are the next most important, as expected. The next most important feature seems to be wind.

### Results:
```
par                     64.544%
solar radiation         5.584%
total solar radiation   4.705%
timestamp (frac)        2.832%
timestamp (int)         2.331%
time of max wind speed  1.975%
wind direction          1.812%
barometric pressure     1.612%
pan                     1.295%
soil temp 20cm          1.244%
soil temp 5cm           1.177%
water temp              1.121%
soil moisture           1.089%
wind speed              0.973%
time of max rainfall    0.898%
humidity                0.876%
wind direction stddev   0.754%
battery voltage         0.750%
air temp                0.742%
vapor pressure deficit  0.716%
soil temp 10cm          0.650%
dewpoint                0.507%
vapor pressure          0.499%
max wind speed          0.498%
time of max rainfall 2  0.295%
total par               0.254%
rainfall                0.084%
max rainfall            0.084%
rainfall 2              0.059%
max rainfall 2          0.016%
fuel temp               0.014%
fuel moisture           0.011%
soil temp 2cm           0.000%
soil temp a             0.000%
soil temp b             0.000%
evap                    0.000%
net radiation           0.000%
total net radiation     0.000%
leaf wetness            0.000%
wetness frequency       0.000%
```


## Part 2

This experiment trains an XGBoost regressor for 24 hour predictions using three different feature sets: solar radiation and wind; solar radiation, temperature, humidity, and rainfall (identified by Cameron as a good feature set); and solar radiation alone.

The results to indicate that, in addition to solar radiation, wind features improve accuracy while temperature, humidity, and rainfall together reduce accuracy. The results are not statistically significant.

### Results:
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
