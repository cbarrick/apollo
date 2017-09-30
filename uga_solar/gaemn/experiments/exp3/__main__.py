#!/usr/bin/env python3

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler

from uga_solar.gaemn.data import gaemn15
from uga_solar.gaemn.experiments import core

core.setup()

datasets = {
    gaemn15.DataSet: {
        'path': ['./gaemn15.zip'],
        'years': [range(2003, 2013)],
        'x_features': [('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation')],
        'y_features': [('solar radiation (+4)', )],
        'lag': [4],
    },
}

estimators = {
    Pipeline: {
        'steps': [
            [('predict', RandomForestRegressor())],
            [('stdandardize', StandardScaler()), ('predict', RandomForestRegressor())],
            [('min_max_scale', MinMaxScaler()), ('predict', RandomForestRegressor())],
            [('max_abs_scale', MaxAbsScaler()), ('predict', RandomForestRegressor())],
        ],
    },
}

results = core.percent_split(estimators, datasets, 0.8, nfolds=10)
print(results)
