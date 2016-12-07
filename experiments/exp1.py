#!/usr/bin/env python3
''' Experiment 1

This experiment evaluates the benefits of using delta features and predicted
features to evaluate 1hr solar radiation predictions. We simulate perfect
predictions of some features by using the future values.

The regressor used for all experiments is Random Forest.

The results indicate that predicted features increase accuracy (assuming perfect
predictions) and delta features on't help much.

The results regarding predicted features may not be terribly valid. By
simulating perfect predictions, our conclusions are equivalent to saying that
it is easier to predict current solar radiation than to predict future
radiation. See experiment 4 for a more meaningful inquiry on predicted
features.

Results:
```
trial 1: score=0.9166755592522465, mse=7087.656475723307, mae=37.76623897683379
trial 2: score=0.9175471901287965, mse=7013.514721261111, mae=37.513857615811155
trial 3: score=0.9352959488877691, mse=5503.788357361402, mae=33.373508791808504
trial 4: score=0.9335297691536251, mse=5654.021291632063, mae=33.82115457236394
```
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data import gaemn15

# Trial 1
# -------------------------
# Random Forest using standard features

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
# Random Forest using standard features and deltas

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
print('trial 2: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 3
# -------------------------
# Random Forest using standard and "predicted" features

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4)', 'humidity (+4)', 'rainfall (+4)',),
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
# Random Forest using standard and "predicted" features and deltas

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4)', 'humidity (+4)', 'rainfall (+4)',
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
print('trial 4: score={}, mse={}, mae={}'.format(score, mse, mae))
