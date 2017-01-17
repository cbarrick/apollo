#!/usr/bin/env python3
''' Experiment 10

```
par
solar radiation
total solar radiation
timestamp (frac)
timestamp (int)
time of max wind speed
wind direction
barometric pressure
pan
soil temp 20cm
soil temp 5cm
water temp
soil moisture
wind speed
time of max rainfall
humidity
wind direction stddev
battery voltage
air temp
vapor pressure deficit
soil temp 10cm
dewpoint
vapor pressure
max wind speed
time of max rainfall 2
total par
rainfall
max rainfall
rainfall 2
max rainfall 2
fuel temp
fuel moisture
soil temp 2cm
soil temp a
soil temp b
evap
net radiation
total net radiation
leaf wetness
wetness frequency
```
'''

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import scale as standard_scale

from experiments import core
from data import gaemn15


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

data = gaemn15.DataSet(
    path       = './gaemn15.zip',
    years      = range(2003,2011),
    x_features = features,
    y_features = ('solar radiation (+96)',),
    lag        = 1,
    scale      = standard_scale,
)

model = RandomForestRegressor()
model.fit(data.data, data.target)
rankings = model.feature_importances_
ranked = sorted(features, key=lambda f: rankings[features.index(f)], reverse=True)
print(ranked)
