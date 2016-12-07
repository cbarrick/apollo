#!/usr/bin/env python3
''' Experiment 3

Experiment 1 told us that we can predict some features and use those predicted
features as additional inputs to predict solar radiation.

This experiment tries to see which features are most beneficial to predict by
simulating a perfect prediction of those

The results seem to indicate that more predicted inputs are better, but
predicted rainfall is the most useless. This is equivalent to saying that
predicting current solar radiation using current features and some old features
is easier than predicting into the future. See experiment 4 for a more
meaningful inquiry on predicted features.

Results:
```
trial 1: score=0.9169741324454534, mse=7062.2595548760755, mae=37.71840227909333
trial 2: score=0.931137190556372, mse=5857.536311190773, mae=34.57914493548372
trial 3: score=0.9314721850702149, mse=5829.0413580406175, mae=34.39568956280536
trial 4: score=0.9185894765891183, mse=6924.856839926844, mae=37.43386435929828
trial 5: score=0.934550832242553, mse=5567.168690529469, mae=33.58040939438432
trial 6: score=0.9320699806355142, mse=5778.192296570428, mae=34.254505339321305
trial 7: score=0.9319630158666116, mse=5787.290821927253, mae=34.42924780701948
trial 8: score=0.9359582451024963, mse=5447.452809075291, mae=33.32399635409552
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
# Random Forest using predicted air temp.

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4)',),
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
# Random Forest using predicted humidity.

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'humidity (+4)',),
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
# Random Forest using predicted rainfall.

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'rainfall (+4)',),
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
# Random Forest using predicted humidity and rainfall.

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'humidity (+4)', 'rainfall (+4)'),
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

# Trial 8
# -------------------------
# Random Forest using predicted air temp, humidity, and rainfall

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4)', 'humidity (+4)', 'rainfall (+4)'),
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
print('trial 8: score={}, mse={}, mae={}'.format(score, mse, mae))
