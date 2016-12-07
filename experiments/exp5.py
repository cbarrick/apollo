#!/usr/bin/env python3
''' Experiment 5

This experiment repeates experiment 4 as a six hour prediction problem.

The results indicate that if we can predict features like air temperature with
a reasonable accuracy, then we can use those predicted features as additional
inputs to improve solar radiation prediction.

Results:
```
trial 0: score=0.8293957684478781, mse=14517.175145864043, mae=61.40987584415872
trial 1: score=0.9206079022689934, mse=6755.688164783059, mae=40.38353344184268
trial 2: score=0.9138364911542214, mse=7331.885837276093, mae=42.022838563784916
trial 3: score=0.9088148072834452, mse=7759.19448968163, mae=43.62880297530456
trial 4: score=0.8935472240743068, mse=9058.354407842282, mae=47.88028233015753
trial 5: score=0.8745646233265914, mse=10673.635208743186, mae=52.78455887887884
trial 6: score=0.8539487102205908, mse=12427.898972478068, mae=57.621579969761335
trial 7: score=0.8372532414756743, mse=13848.56152994931, mae=60.31676184749922
```
'''

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data import gaemn15

# Baseline
# -------------------------

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'),
    'y_features' : ('solar radiation (+24)',),
    'window'     : 24,
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
print('trial 0: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 1
# -------------------------

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+24,noise=0)', 'humidity (+24,noise=0)', 'rainfall (+24,noise=0)'),
    'y_features' : ('solar radiation (+24)',),
    'window'     : 24,
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

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+24,noise=0.1)', 'humidity (+24,noise=0.1)', 'rainfall (+24,noise=0.1)'),
    'y_features' : ('solar radiation (+24)',),
    'window'     : 24,
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

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+24,noise=0.2)', 'humidity (+24,noise=0.2)', 'rainfall (+24,noise=0.2)'),
    'y_features' : ('solar radiation (+24)',),
    'window'     : 24,
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

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+24,noise=0.5)', 'humidity (+24,noise=0.5)', 'rainfall (+24,noise=0.5)'),
    'y_features' : ('solar radiation (+24)',),
    'window'     : 24,
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
print('trial 4: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 5
# -------------------------

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+24,noise=1)', 'humidity (+24,noise=1)', 'rainfall (+24,noise=1)'),
    'y_features' : ('solar radiation (+24)',),
    'window'     : 24,
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
print('trial 5: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 6
# -------------------------

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+24,noise=2)', 'humidity (+24,noise=2)', 'rainfall (+24,noise=2)'),
    'y_features' : ('solar radiation (+24)',),
    'window'     : 24,
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
print('trial 6: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 7
# -------------------------

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+24,noise=3)', 'humidity (+24,noise=3)', 'rainfall (+24,noise=3)'),
    'y_features' : ('solar radiation (+24)',),
    'window'     : 24,
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
print('trial 7: score={}, mse={}, mae={}'.format(score, mse, mae))
