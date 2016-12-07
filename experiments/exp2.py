#!/usr/bin/env python3
''' Experiment 2

This experiment is designed to test the various timestamp features of the
dataset. We have the following features:

- 'day' gives the day of the year, between 1 and 366.
- 'time' gives the time of day like 145 for 1:45AM and 200 for 2:00AM.
- 'timestamp' gives a floating point timestamp. The integer part is the day of
  the year and the fractional part is the time of day.

The problem with 'time' is that the value increases by 15 within the hour, e.g.
from 100 (1:00AM) to 115 (1:15AM), and by 55 between hours, e.g. 145 (1:45AM) to
200 (2:00AM). The problem with 'timestamp' is that it combines two periodic
signals, the day of year and time of day. Ideally, we'd like separate uniform
signals for day of year and time of day.

This experiment uses Random Forest to compare the use of the features 'day' and
'time' together vs. 'timestamp' alone vs. 'day' and 'timestamp' together vs. a
realization of the ideal signals.

The results show that the ideal signal is only slightly better than using 'day'
and 'time' together, which is better than using 'timestamp' alone or 'day' and
'timestamp' together. The differences may not be significant. This experiment
applies to Random Forest. My gut tells me that the story might be different for
neural nets.

Results:
```
trial 1: score=0.9177848209609276, mse=6993.301615825644, mae=37.40623534652022
trial 2: score=0.8955900553974068, mse=8881.209562902868, mae=43.93510424183818
trial 3: score=0.8963773857259806, mse=8814.23849352909, mae=43.760859856823316
trial 4: score=0.9172906988077565, mse=7035.332117888344, mae=37.54017640960852
```
'''

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data import gaemn15

# Trial 1
# -------------------------
# Random Forest using day and time

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'),
    'y_features' : ('solar radiation (+4)',),
    'window'     : 4,
}

griffin_train = gaemn15.DataSet(**data_params, years=range(2003,2011))
griffin_test  = gaemn15.DataSet(**data_params, years=range(2011,2013))
train = griffin_train.data, griffin_train.target
test  = griffin_test.data, griffin_test.target

rand_forest = RandomForestRegressor()
rand_forest.fit(train[0], train[1])
pred = rand_forest.predict(test[0])

score = r2_score(test[1], pred)
mse = mean_squared_error(test[1], pred)
mae = mean_absolute_error(test[1], pred)
print('trial 1: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 2
# -------------------------
# Random Forest using day and timestamp

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'timestamp', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'),
    'y_features' : ('solar radiation (+4)',),
    'window'     : 4,
}

griffin_train = gaemn15.DataSet(**data_params, years=range(2003,2011))
griffin_test  = gaemn15.DataSet(**data_params, years=range(2011,2013))
train = griffin_train.data, griffin_train.target
test  = griffin_test.data, griffin_test.target

rand_forest = RandomForestRegressor()
rand_forest.fit(train[0], train[1])
pred = rand_forest.predict(test[0])

score = r2_score(test[1], pred)
mse = mean_squared_error(test[1], pred)
mae = mean_absolute_error(test[1], pred)
print('trial 2: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 3
# -------------------------
# Random Forest using only timestamp

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('timestamp', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'),
    'y_features' : ('solar radiation (+4)',),
    'window'     : 4,
}

griffin_train = gaemn15.DataSet(**data_params, years=range(2003,2011))
griffin_test  = gaemn15.DataSet(**data_params, years=range(2011,2013))
train = griffin_train.data, griffin_train.target
test  = griffin_test.data, griffin_test.target

rand_forest = RandomForestRegressor()
rand_forest.fit(train[0], train[1])
pred = rand_forest.predict(test[0])

score = r2_score(test[1], pred)
mse = mean_squared_error(test[1], pred)
mae = mean_absolute_error(test[1], pred)
print('trial 3: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 4
# -------------------------
# Random Forest using ideal signal

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('timestamp', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'),
    'y_features' : ('solar radiation (+4)',),
    'window'     : 4,
}

griffin_train = gaemn15.DataSet(**data_params, years=range(2003,2011))
griffin_test  = gaemn15.DataSet(**data_params, years=range(2011,2013))
train = griffin_train.data, griffin_train.target
test  = griffin_test.data, griffin_test.target

# build the ideal signal by hand since the DataSet class doesn't yet know how
def ideal(data):
    time, day = np.modf(data[:, 0:1])
    return np.concatenate((time, day, data[:, 1:]), axis=1)
train = (ideal(train[0]), train[1])
test  = (ideal(test[0]),  test[1])

rand_forest = RandomForestRegressor()
rand_forest.fit(train[0], train[1])
pred = rand_forest.predict(test[0])

score = r2_score(test[1], pred)
mse = mean_squared_error(test[1], pred)
mae = mean_absolute_error(test[1], pred)
print('trial 4: score={}, mse={}, mae={}'.format(score, mse, mae))
