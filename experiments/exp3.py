#!/usr/bin/env python3
''' Experiment 3

Experiment 1 told us that we can predict some attributes and use those
predicted attributes as additional inputs to predict solar radiation.

This experiment tries to see which attributes are most beneficial to predict.

The results seem to indicate that more predicted inputs are better.

Results:
```
trial 1: score=0.9176105426548267, mse=7008.125894917468, mae=37.52384597524507
trial 2: score=0.9313408240904069, mse=5840.2150483869145, mae=34.638420749523114
trial 3: score=0.9311128246960598, mse=5859.608894529812, mae=34.443706510486216
trial 4: score=0.9187253808577106, mse=6913.296693099306, mae=37.30875120719477
trial 5: score=0.9349190074801536, mse=5535.851353340485, mae=33.50560613669589
trial 6: score=0.931742130471653, mse=5806.079544490784, mae=34.27554559813957
trial 7: score=0.9315644325527799, mse=5821.19469910954, mae=34.47870600418086
trial 8: score=0.935421902286593, mse=5493.074641015955, mae=33.44495054346616
```
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data import gaemn15

# Trial 1
# -------------------------
# Random Forest using no predicted features.

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'),
    'y_features' : ('solar radiation (+4)',),
    'window'     : 4,
    'deltas'     : False,
}

griffin_train = gaemn15.DataSet(**data_params, years=range(2003,2011))
griffin_test  = gaemn15.DataSet(**data_params, years=range(2011,2013))
train = griffin_train.data, griffin_train.target[:, 3::4].ravel()
test  = griffin_test.data, griffin_test.target[:, 3::4].ravel()

rand_forest = RandomForestRegressor()
rand_forest.fit(train[0], train[1])
pred = rand_forest.predict(test[0])

score = r2_score(test[1], pred)
mse = mean_squared_error(test[1], pred)
mae = mean_absolute_error(test[1], pred)
print('trial 1: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 2
# -------------------------
# Random Forest using predicted air temp.

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4)',),
    'y_features' : ('solar radiation (+4)',),
    'window'     : 4,
    'deltas'     : False,
}

griffin_train = gaemn15.DataSet(**data_params, years=range(2003,2011))
griffin_test  = gaemn15.DataSet(**data_params, years=range(2011,2013))
train = griffin_train.data, griffin_train.target[:, 3::4].ravel()
test  = griffin_test.data, griffin_test.target[:, 3::4].ravel()

rand_forest = RandomForestRegressor()
rand_forest.fit(train[0], train[1])
pred = rand_forest.predict(test[0])

score = r2_score(test[1], pred)
mse = mean_squared_error(test[1], pred)
mae = mean_absolute_error(test[1], pred)
print('trial 2: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 3
# -------------------------
# Random Forest using predicted humidity.

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'humidity (+4)',),
    'y_features' : ('solar radiation (+4)',),
    'window'     : 4,
    'deltas'     : False,
}

griffin_train = gaemn15.DataSet(**data_params, years=range(2003,2011))
griffin_test  = gaemn15.DataSet(**data_params, years=range(2011,2013))
train = griffin_train.data, griffin_train.target[:, 3::4].ravel()
test  = griffin_test.data, griffin_test.target[:, 3::4].ravel()

rand_forest = RandomForestRegressor()
rand_forest.fit(train[0], train[1])
pred = rand_forest.predict(test[0])

score = r2_score(test[1], pred)
mse = mean_squared_error(test[1], pred)
mae = mean_absolute_error(test[1], pred)
print('trial 3: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 4
# -------------------------
# Random Forest using predicted rainfall.

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'rainfall (+4)',),
    'y_features' : ('solar radiation (+4)',),
    'window'     : 4,
    'deltas'     : False,
}

griffin_train = gaemn15.DataSet(**data_params, years=range(2003,2011))
griffin_test  = gaemn15.DataSet(**data_params, years=range(2011,2013))
train = griffin_train.data, griffin_train.target[:, 3::4].ravel()
test  = griffin_test.data, griffin_test.target[:, 3::4].ravel()

rand_forest = RandomForestRegressor()
rand_forest.fit(train[0], train[1])
pred = rand_forest.predict(test[0])

score = r2_score(test[1], pred)
mse = mean_squared_error(test[1], pred)
mae = mean_absolute_error(test[1], pred)
print('trial 4: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 5
# -------------------------
# Random Forest using predicted air temp and humidity.

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4)', 'humidity (+4)'),
    'y_features' : ('solar radiation (+4)',),
    'window'     : 4,
    'deltas'     : False,
}

griffin_train = gaemn15.DataSet(**data_params, years=range(2003,2011))
griffin_test  = gaemn15.DataSet(**data_params, years=range(2011,2013))
train = griffin_train.data, griffin_train.target[:, 3::4].ravel()
test  = griffin_test.data, griffin_test.target[:, 3::4].ravel()

rand_forest = RandomForestRegressor()
rand_forest.fit(train[0], train[1])
pred = rand_forest.predict(test[0])

score = r2_score(test[1], pred)
mse = mean_squared_error(test[1], pred)
mae = mean_absolute_error(test[1], pred)
print('trial 5: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 6
# -------------------------
# Random Forest using predicted humidity and rainfall.

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'humidity (+4)', 'rainfall (+4)'),
    'y_features' : ('solar radiation (+4)',),
    'window'     : 4,
    'deltas'     : False,
}

griffin_train = gaemn15.DataSet(**data_params, years=range(2003,2011))
griffin_test  = gaemn15.DataSet(**data_params, years=range(2011,2013))
train = griffin_train.data, griffin_train.target[:, 3::4].ravel()
test  = griffin_test.data, griffin_test.target[:, 3::4].ravel()

rand_forest = RandomForestRegressor()
rand_forest.fit(train[0], train[1])
pred = rand_forest.predict(test[0])

score = r2_score(test[1], pred)
mse = mean_squared_error(test[1], pred)
mae = mean_absolute_error(test[1], pred)
print('trial 6: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 7
# -------------------------
# Random Forest using predicted air temp and rainfall

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4)', 'rainfall (+4)'),
    'y_features' : ('solar radiation (+4)',),
    'window'     : 4,
    'deltas'     : False,
}

griffin_train = gaemn15.DataSet(**data_params, years=range(2003,2011))
griffin_test  = gaemn15.DataSet(**data_params, years=range(2011,2013))
train = griffin_train.data, griffin_train.target[:, 3::4].ravel()
test  = griffin_test.data, griffin_test.target[:, 3::4].ravel()

rand_forest = RandomForestRegressor()
rand_forest.fit(train[0], train[1])
pred = rand_forest.predict(test[0])

score = r2_score(test[1], pred)
mse = mean_squared_error(test[1], pred)
mae = mean_absolute_error(test[1], pred)
print('trial 7: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 8
# -------------------------
# Random Forest using predicted air temp, humidity, and rainfall

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4)', 'humidity (+4)', 'rainfall (+4)'),
    'y_features' : ('solar radiation (+4)',),
    'window'     : 4,
    'deltas'     : False,
}

griffin_train = gaemn15.DataSet(**data_params, years=range(2003,2011))
griffin_test  = gaemn15.DataSet(**data_params, years=range(2011,2013))
train = griffin_train.data, griffin_train.target[:, 3::4].ravel()
test  = griffin_test.data, griffin_test.target[:, 3::4].ravel()

rand_forest = RandomForestRegressor()
rand_forest.fit(train[0], train[1])
pred = rand_forest.predict(test[0])

score = r2_score(test[1], pred)
mse = mean_squared_error(test[1], pred)
mae = mean_absolute_error(test[1], pred)
print('trial 8: score={}, mse={}, mae={}'.format(score, mse, mae))
