#!/usr/bin/env python3
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, \
 GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import scale as standard_scale

from uga_solar.data import gaemn
from .. import core

core.setup()

datasets = {
    gaemn.GaemnLoader: {
        'path': ['./gaemn.zip'],
        'years': [range(2003, 2013)],
        'x_features': [('day', 'time', 'air temp', 'humidity', 'rainfall', 'solar radiation')],
        'y_features': [('solar radiation (+24)', )],
        'lag': [4],
        'scale': [standard_scale],
    },
}  # yapf: disable

estimators = {
    ExtraTreesRegressor: {},
    RandomForestRegressor: {},
    BaggingRegressor: {},
    GradientBoostingRegressor: {},
    XGBRegressor: {},
}

results = core.percent_split(estimators, datasets, 0.8, nfolds=10)
print(results)
