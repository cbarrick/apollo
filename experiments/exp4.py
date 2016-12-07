#!/usr/bin/env python3
''' Experiment 4

Experiment 3 told us that if we have perfect predictions of certain features,
it becomes easier to predict solar radiation. This is equivalent to saying that
predicting current solar radiation is easier than predicting future radiation.

This experiment attempts to test that idea more meaningfully by adding noise to
the simulated predictions. Each successive trial adds more noise to the
predicted features. The noise is drawn from a normal distribution centered
around 0 with a standard deviation equal to some multiple of the standard
deviation of the prefect prediction.

The results indicate that these predicted features are useful for predicting
solar radiation one hour in the future. The benefits of the predicted features
decrease as the noise increases. In these one hour prediction problems, the
benefits of predicted featuers are lost when the standard deviation of the
noise is greater than or equal to the standard deviation of the true data.
However, experiment 5 suggests that the noise can be more varied but still
beneficial for longer term prediction problems.

Results:
```
trial 0: score=0.9171712487938263, mse=7045.492650106264, mae=37.59239959936898
trial 1: score=0.9358518464870593, mse=5456.503176877847, mae=33.34116157927068
trial 2: score=0.9294350672456501, mse=6002.320544933664, mae=35.006023967416006
trial 3: score=0.9254018651955405, mse=6345.388561614356, mae=35.98659972476165
trial 4: score=0.9222170474587106, mse=6616.292198696966, mae=36.808064354310815
trial 5: score=0.9200240868517232, mse=6802.827521438258, mae=37.444581782840764
trial 6: score=0.9181009127009734, mse=6966.414550660211, mae=37.76590432119329
trial 7: score=0.919013601932766, mse=6888.780333304697, mae=37.57565501779986
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
print('trial 0: score={}, mse={}, mae={}'.format(score, mse, mae))

# Trial 1
# -------------------------

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4,noise=0)', 'humidity (+4,noise=0)', 'rainfall (+4,noise=0)'),
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

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4,noise=0.1)', 'humidity (+4,noise=0.1)', 'rainfall (+4,noise=0.1)'),
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

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4,noise=0.2)', 'humidity (+4,noise=0.2)', 'rainfall (+4,noise=0.2)'),
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

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4,noise=0.5)', 'humidity (+4,noise=0.5)', 'rainfall (+4,noise=0.5)'),
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

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4,noise=1)', 'humidity (+4,noise=1)', 'rainfall (+4,noise=1)'),
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

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4,noise=2)', 'humidity (+4,noise=2)', 'rainfall (+4,noise=2)'),
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

data_params = {
    'path'       : './gaemn15.zip',
    'x_features' : ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                    'air temp (+4,noise=3)', 'humidity (+4,noise=3)', 'rainfall (+4,noise=3)'),
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
