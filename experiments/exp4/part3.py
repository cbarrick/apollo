#!/usr/bin/env python3
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, \
 GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import scale as standard_scale

from experiments import core
from data import gaemn15

core.setup()

datasets = {
    gaemn15.DataSet: {
        'path': ['./gaemn15.zip'],
        'years': [range(2003, 2013)],
        'x_features': [('timestamp (int)', 'timestamp (frac)', 'solar radiation')],
        'y_features': [('solar radiation (+96)', )],
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
