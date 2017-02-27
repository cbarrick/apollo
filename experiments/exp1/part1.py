#!/usr/bin/env python3
from sklearn.ensemble import RandomForestRegressor

from experiments import core
from data import gaemn15

core.setup()

datasets = {
    gaemn15.DataSet: {
        'path': ['./gaemn15.zip'],
        'years': [range(2003,2013)],
        'x_features': [
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'humidity (+4)', 'rainfall (+4)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'humidity (+4)', 'rainfall (+4)', 'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'),
        ],
        'y_features': [('solar radiation (+4)',)],
        'lag': [4],
    }
} # yapf: disable

estimators = {
    RandomForestRegressor: {},
}

results = core.percent_split(estimators, datasets, 0.8, nfolds=10)
print(results)
