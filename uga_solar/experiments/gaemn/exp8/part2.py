#!/usr/bin/env python3
import logging

from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, \
 GradientBoostingRegressor, RandomForestRegressor
# from xgboost import XGBRegressor

from sklearn.preprocessing import scale as standard_scale

from uga_solar.data import gaemn
from .. import core

core.setup()

datasets = {
    gaemn.GaemnLoader: {
        'path': ['./gaemn.zip'],
        'years': [range(2003, 2013)],
        'x_features': [
            ('timestamp (int)', 'timestamp (frac)', 'solar radiation'),
            ('timestamp (int)', 'timestamp (frac)', 'solar radiation', 'wind speed', 'wind direction'),
            ('timestamp (int)', 'timestamp (frac)', 'solar radiation', 'wind speed', 'wind direction', 'barometric pressure'),
        ],
        'y_features': [('solar radiation (+4)', )],
        'lag': [24],
    },
} # yapf: disable

estimators = {
    RandomForestRegressor: {},
}

results = core.percent_split(estimators, datasets, 0.8, nfolds=10)
print(results)
