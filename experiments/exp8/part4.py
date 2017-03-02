#!/usr/bin/env python3
import logging

from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, \
 GradientBoostingRegressor, RandomForestRegressor
# from xgboost import XGBRegressor

from sklearn.preprocessing import scale as standard_scale

from experiments import core
from data import gaemn15

core.setup()

datasets = {
    gaemn15.DataSet: {
        'path': ['./gaemn15.zip'],
        'years': [range(2003, 2013)],
        'x_features': [('timestamp (int)', 'timestamp (frac)', 'solar radiation')],
        'y_features': [('solar radiation (+4)', )],
        'lag': [24],
    },
}

estimators = {
    ExtraTreesRegressor: {
        'n_estimators': [500, 750, 1000, 1500, 2000],
        'n_jobs': [-1],
    },
}

results = core.percent_split(estimators, datasets, 0.8, nfolds=10)
print(results)
