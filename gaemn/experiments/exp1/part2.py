#!/usr/bin/env python3
from sklearn.ensemble import RandomForestRegressor

from gaemn.data import gaemn15
from gaemn.experiments import core

core.setup()

datasets = {
    gaemn15.DataSet: {
        'path': ['./gaemn15.zip'],
        'years': [range(2003,2013)],
        'x_features' : [
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'humidity (+4)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'rainfall (+4)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'humidity (+4)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'humidity (+4)', 'rainfall (+4)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'rainfall (+4)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4)', 'humidity (+4)', 'rainfall (+4)'),
        ],
        'y_features': [('solar radiation (+4)',)],
        'lag': [4],
    }
} # yapf: disable

estimators = {
    RandomForestRegressor: {},
}

results = core.percent_split(estimators, datasets, 0.8)
print(results)
