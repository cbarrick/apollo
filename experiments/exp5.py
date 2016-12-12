#!/usr/bin/env python3
''' Experiment 5

This experiment repeates experiment 4 as a six hour prediction problem.

The results indicate that if we can predict features like air temperature with
a reasonable accuracy, then we can use those predicted features as additional
inputs to improve solar radiation prediction.

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
        y_features = ('solar radiation (+24)',),
        window     = 24,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+24,noise=0)', 'humidity (+24,noise=0)', 'rainfall (+24,noise=0)'),
        y_features = ('solar radiation (+24)',),
        window     = 24,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+24,noise=0.1)', 'humidity (+24,noise=0.1)', 'rainfall (+24,noise=0.1)'),
        y_features = ('solar radiation (+24)',),
        window     = 24,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+24,noise=0.2)', 'humidity (+24,noise=0.2)', 'rainfall (+24,noise=0.2)'),
        y_features = ('solar radiation (+24)',),
        window     = 24,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+24,noise=0.5)', 'humidity (+24,noise=0.5)', 'rainfall (+24,noise=0.5)'),
        y_features = ('solar radiation (+24)',),
        window     = 24,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+24,noise=1)', 'humidity (+24,noise=1)', 'rainfall (+24,noise=1)'),
        y_features = ('solar radiation (+24)',),
        window     = 24,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+24,noise=2)', 'humidity (+24,noise=2)', 'rainfall (+24,noise=2)'),
        y_features = ('solar radiation (+24)',),
        window     = 24,
    ),
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation',
                      'air temp (+24,noise=3)', 'humidity (+24,noise=3)', 'rainfall (+24,noise=3)'),
        y_features = ('solar radiation (+24)',),
        window     = 24,
    ),
]


estimators = {
    RandomForestRegressor(): {},
}

results = core.compare(estimators, datasets, split=0.8)
print(results)
