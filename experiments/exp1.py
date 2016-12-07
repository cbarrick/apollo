#!/usr/bin/env python3
''' Experiment 1

This experiment evaluates the benefits of using delta features and predicted
auxiliary features to evaluate 1hr solar radiation predictions. For the trials
using predicted features, we use the real future vaules to simulate perfect
prediction.

The regressor used for all experiments is Random Forest.

The results indicate that both predicted features and delta features increase
accuracy (assuming perfect predictions) and perfect predictions are more useful
than deltas. The best result combines predictions and deltas.

Results:
```
trial 1: score=0.9161372024960723, mse=7133.449615346033, mae=37.923192402365856
trial 2: score=0.9250815014810756, mse=6372.6868893793935, mae=34.89397077718892
trial 3: score=0.9356299286867611, mse=5475.379716825159, mae=33.32263329790161
trial 4: score=0.9607142960857806, mse=3341.704588633028, mae=25.370811135824805
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
# Random Forest using standard features and deltas

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'),
    'y_features' : ('solar radiation (+4)',),
    'window'     : 4,
    'deltas'     : True,
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
# Random Forest using standard and "predicted" features

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4)', 'humidity (+4)', 'rainfall (+4)',),
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
# Random Forest using standard and "predicted" features and deltas

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4)', 'humidity (+4)', 'rainfall (+4)',),
    'y_features' : ('solar radiation (+4)',),
    'window'     : 4,
    'deltas'     : True,
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
