#!/usr/bin/env python3
from sklearn.ensemble import RandomForestRegressor

from uga_solar.data import gaemn
from .. import core

core.setup()

datasets = {
    gaemn.GaemnLoader: [{
        'path': ['./gaemn.zip'],
        'years': [range(2003,2013)],
        'x_features': [
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation')],
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=0.0)', 'humidity (+4,noise=0.0)', 'rainfall (+4,noise=0.0)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=0.1)', 'humidity (+4,noise=0.1)', 'rainfall (+4,noise=0.1)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=0.2)', 'humidity (+4,noise=0.2)', 'rainfall (+4,noise=0.2)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=0.5)', 'humidity (+4,noise=0.5)', 'rainfall (+4,noise=0.5)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=1.0)', 'humidity (+4,noise=1.0)', 'rainfall (+4,noise=1.0)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=2.0)', 'humidity (+4,noise=2.0)', 'rainfall (+4,noise=2.0)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+4,noise=3.0)', 'humidity (+4,noise=3.0)', 'rainfall (+4,noise=3.0)'),
        ],
        'y_features': [('solar radiation (+4)',)],
        'lag': [4],
    }],
} # yapf: disable

estimators = {
    RandomForestRegressor: {},
}

results = core.percent_split(estimators, datasets, 0.8)
print(results)
