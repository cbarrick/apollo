#!/usr/bin/env python3
import logging
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import scale as standard_scale

from gaemn.data import gaemn15
from gaemn.experiments import core

from experiments.exp7.map import GeorgiaMap

core.setup()

estimators = {
    RandomForestRegressor: {},
}

train = gaemn15.DataSet(
    path='./gaemn15.zip',
    city='GRIFFIN',
    years=range(2003, 2011),
    x_features=('day', 'time', 'solar radiation'),
    y_features=('solar radiation (+96)', ),
    lag=96,
)

test = {
    gaemn15.DataSet: {
        'path': ['./gaemn15.zip'],
        'city': [
            'ALAPAHA', 'ALMA', 'ALPHARET', 'ARLINGT', 'ATLANTA', 'ATTAPUL', 'BLAIRSVI', 'BRUNSW',
            'BYRON', 'CAIRO', 'CALHOUN', 'CAMILLA', 'CORDELE', 'COVING', 'DAHLON', 'DALLAS',
            'DAWSON', 'DEARING', 'DIXIE', 'DUBLIN', 'DUNWOODY', 'EATONTON', 'ELBERTON', 'ELLIJAY',
            'FLOYD', 'FTVALLEY', 'GAINES', 'GEORGETO', 'GRIFFIN', 'HOMERV', 'JONESB', 'JVILLE',
            'LAFAYET', 'MCRAE', 'MIDVILLE', 'NAHUNTA', 'NEWTON', 'PLAINS', 'SAVANNAH', 'SHELLMAN',
            'SKIDAWAY', 'STATES', 'TIFTON', 'VALDOSTA', 'VIDALIA', 'WANSLEY', 'WATUGA'
        ],
        'years': [range(2011, 2013)],
        'x_features': [('day', 'time', 'solar radiation')],
        'y_features': [('solar radiation (+96)', )],
        'lag': [96],
    },
}

results = core.train_test(
    estimators,
    train,
    test,
    nfolds=10,
    metric=mean_absolute_error,
)

print(results.summary())

t = results.trials()
map_data = {k[1]: v for k, v in t.items()}

plt.figure(figsize=(10, 10))
m = GeorgiaMap(map_data, resolution='i')
m.draw(200, 200)
m.plot('GRIFFIN', 'ro')
plt.title('Mean Absolute Error (2011-2012)')
plt.show()
