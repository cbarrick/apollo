#!/usr/bin/env python3
import logging

from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, \
 GradientBoostingRegressor, RandomForestRegressor
# from xgboost import XGBRegressor

from sklearn.preprocessing import scale as standard_scale

from gaemn.data import gaemn15
from gaemn.experiments import core

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
    RandomForestRegressor: {
        'n_estimators': [10, 25, 50, 100, 200, 500],
        'n_jobs': [-1],
    },
    ExtraTreesRegressor: {
        'n_estimators': [10, 25, 50, 100, 200, 500],
        'n_jobs': [-1],
    },
    GradientBoostingRegressor: {
        'n_estimators': [10, 25, 50, 100, 200, 500],
    },
    BaggingRegressor: {
        'n_estimators': [10, 25, 50, 100, 200, 500],
        'n_jobs': [-1],
    },
}

results = core.percent_split(estimators, datasets, 0.8, nfolds=10)
print(results)
