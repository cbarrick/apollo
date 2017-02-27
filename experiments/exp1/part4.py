#!/usr/bin/env python3
from sklearn.ensemble import RandomForestRegressor

from experiments import core
from data import gaemn15

core.setup()

datasets = {
    gaemn15.DataSet: [{
        'path': ['./gaemn15.zip'],
        'years': [range(2003,2013)],
        'x_features' : [
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation')],
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=0)', 'humidity (+24,noise=0)', 'rainfall (+24,noise=0)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=0.1)', 'humidity (+24,noise=0.1)', 'rainfall (+24,noise=0.1)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=0.2)', 'humidity (+24,noise=0.2)', 'rainfall (+24,noise=0.2)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=0.5)', 'humidity (+24,noise=0.5)', 'rainfall (+24,noise=0.5)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=1)', 'humidity (+24,noise=1)', 'rainfall (+24,noise=1)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=2)', 'humidity (+24,noise=2)', 'rainfall (+24,noise=2)'),
            ('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation', 'air temp (+24,noise=3)', 'humidity (+24,noise=3)', 'rainfall (+24,noise=3)'),
        ],
        'y_features': [('solar radiation (+24)',)],
        'lag': [24],
    }],
} # yapf: disable


estimators = {
    RandomForestRegressor: {},
}

results = core.percent_split(estimators, datasets, 0.8)
print(results)
