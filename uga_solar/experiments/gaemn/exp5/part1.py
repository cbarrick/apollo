#!/usr/bin/env python3
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import scale as standard_scale

from uga_solar.data import gaemn
from .. import core


core.setup()

features = (
    'timestamp (int)',
    'timestamp (frac)',
    'air temp',
    'humidity',
    'dewpoint',
    'vapor pressure',
    'vapor pressure deficit',
    'barometric pressure',
    'wind speed',
    'wind direction',
    'wind direction stddev',
    'max wind speed',
    'time of max wind speed',
    'soil temp 2cm',
    'soil temp 5cm',
    'soil temp 10cm',
    'soil temp 20cm',
    'soil temp a',
    'soil temp b',
    'soil moisture',
    'pan',
    'evap',
    'water temp',
    'solar radiation',
    'total solar radiation',
    'par',
    'total par',
    'net radiation',
    'total net radiation',
    'rainfall',
    'rainfall 2',
    'max rainfall',
    'time of max rainfall',
    'max rainfall 2',
    'time of max rainfall 2',
    'leaf wetness',
    'wetness frequency',
    'battery voltage',
    'fuel temp',
    'fuel moisture',
)

data = gaemn.GaemnLoader(
    path       = './gaemn.zip',
    years      = range(2003,2011),
    x_features = features,
    y_features = ('solar radiation (+96)',),
    lag        = 1,
    scale      = standard_scale,
) # yapf: disable

model = RandomForestRegressor()
model.fit(data.data, data.target)
rankings = model.feature_importances_
ranked = sorted(features, key=lambda f: rankings[features.index(f)], reverse=True)
for f in ranked:
    print(f'{f:24}{rankings[features.index(f)]:.03%}')
