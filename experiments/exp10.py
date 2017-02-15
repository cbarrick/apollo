#!/usr/bin/env python3
''' Experiment 10

```
par                     64.544%
solar radiation         5.584%
total solar radiation   4.705%
timestamp (frac)        2.832%
timestamp (int)         2.331%
time of max wind speed  1.975%
wind direction          1.812%
barometric pressure     1.612%
pan                     1.295%
soil temp 20cm          1.244%
soil temp 5cm           1.177%
water temp              1.121%
soil moisture           1.089%
wind speed              0.973%
time of max rainfall    0.898%
humidity                0.876%
wind direction stddev   0.754%
battery voltage         0.750%
air temp                0.742%
vapor pressure deficit  0.716%
soil temp 10cm          0.650%
dewpoint                0.507%
vapor pressure          0.499%
max wind speed          0.498%
time of max rainfall 2  0.295%
total par               0.254%
rainfall                0.084%
max rainfall            0.084%
rainfall 2              0.059%
max rainfall 2          0.016%
fuel temp               0.014%
fuel moisture           0.011%
soil temp 2cm           0.000%
soil temp a             0.000%
soil temp b             0.000%
evap                    0.000%
net radiation           0.000%
total net radiation     0.000%
leaf wetness            0.000%
wetness frequency       0.000%

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
) # yapf: disable

model = RandomForestRegressor()
model.fit(data.data, data.target)
rankings = model.feature_importances_
ranked = sorted(features, key=lambda f: rankings[features.index(f)], reverse=True)
for f in ranked:
    print(f'{f:24}{rankings[features.index(f)]:.03%}')
