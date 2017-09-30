#!/usr/bin/env python3
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, \
    ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, \
    Lars, Lasso, LinearRegression, OrthogonalMatchingPursuit, \
    PassiveAggressiveRegressor, RANSACRegressor, TheilSenRegressor
from xgboost import XGBRegressor

from sklearn.preprocessing import scale as standard_scale

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
        'scale': [standard_scale],
    },
}  # yapf: disable

estimators = {
    AdaBoostRegressor: {},
    BaggingRegressor: {},
    ExtraTreesRegressor: {},
    GradientBoostingRegressor: {},
    RandomForestRegressor: {},
    BayesianRidge: {},
    ElasticNet: {},
    HuberRegressor: {},
    Lars: {},
    Lasso: {},
    LinearRegression: {},
    OrthogonalMatchingPursuit: {},
    PassiveAggressiveRegressor: {},
    RANSACRegressor: {},
    TheilSenRegressor: {},
    XGBRegressor: {},
}

results = core.percent_split(estimators, datasets, 0.8, nfolds=10)
print(results)
