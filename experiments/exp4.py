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
```
'''

from sklearn.ensemble import RandomForestRegressor

from experiments import core
from data import gaemn15

core.setup()

datasets = [
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'),
        y_features = ('solar radiation (+4)',),
        window     = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4,noise=0)', 'humidity (+4,noise=0)', 'rainfall (+4,noise=0)'),
        y_features = ('solar radiation (+4)',),
        window     = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4,noise=0.1)', 'humidity (+4,noise=0.1)', 'rainfall (+4,noise=0.1)'),
        y_features = ('solar radiation (+4)',),
        window     = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4,noise=0.2)', 'humidity (+4,noise=0.2)', 'rainfall (+4,noise=0.2)'),
        y_features = ('solar radiation (+4)',),
        window     = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4,noise=0.5)', 'humidity (+4,noise=0.5)', 'rainfall (+4,noise=0.5)'),
        y_features = ('solar radiation (+4)',),
        window     = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4,noise=1)', 'humidity (+4,noise=1)', 'rainfall (+4,noise=1)'),
        y_features = ('solar radiation (+4)',),
        window     = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4,noise=2)', 'humidity (+4,noise=2)', 'rainfall (+4,noise=2)'),
        y_features = ('solar radiation (+4)',),
        window     = 4,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+4,noise=3)', 'humidity (+4,noise=3)', 'rainfall (+4,noise=3)'),
        y_features = ('solar radiation (+4)',),
        window     = 4,
    ),
]

estimators = {
    RandomForestRegressor(): {},
}

results = core.compare(estimators, datasets, split=0.8)
print(results)
