#!/usr/bin/env python3
from sklearn.ensemble import RandomForestRegressor

from uga_solar.gaemn.data import gaemn15
from uga_solar.gaemn.experiments import core

core.setup()

datasets = {
    gaemn15.DataSet: {
        'path': ['./gaemn15.zip'],
        'years': [range(2003,2013)],
        'x_features': [
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'),
            ('day', 'timestamp', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'),
            ('timestamp', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'),
            ('timestamp (int)', 'timestamp (frac)', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (delta)', 'humidity (delta)', 'rainfall (delta)', 'solar radiation (delta)'),
        ],
        'y_features' : [('solar radiation (+4)',)],
        'lag'        : [4],
    },
} # yapf: disable

estimators = {
    RandomForestRegressor: {},
}

results = core.percent_split(estimators, datasets, 0.8)
print(results)