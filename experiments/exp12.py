#!/usr/bin/env python3
'''

Results:
```
```
'''

import tensorflow as tf

from sklearn.preprocessing import scale as standard_scale

from data import gaemn15
from experiments import core
from models.nn import ConvRegressor, MLPRegressor


core.setup()

datasets = [
    gaemn15.DataSet(
        path       = './gaemn15.zip',
        years      = range(2003,2013),
        x_features = ('day', 'time', 'solar radiation'),
        y_features = ('solar radiation (+96)',),
        lag        = 96,
		scale      = standard_scale,
    ),
]

estimators = {
    ConvRegressor(): [{
        'lag': [96],
        'activation': [tf.nn.elu, tf.nn.tanh],
        'initializer': [tf.contrib.layers.xavier_initializer(),
                        tf.contrib.layers.variance_scaling_initializer(2)],
        'regularizer': [tf.contrib.layers.l2_regularizer(0.01), None],
        'optimizer': [tf.train.AdamOptimizer(1e-4)],
    }],
    MLPRegressor(): {
        'layers': [(32,), (64,), (128,), (64,32)],
        'activation': [tf.nn.elu, tf.nn.tanh],
        'initializer': [tf.contrib.layers.xavier_initializer(),
                        tf.contrib.layers.variance_scaling_initializer(2)],
        'regularizer': [tf.contrib.layers.l2_regularizer(0.01), None],
        'optimizer': [tf.train.AdamOptimizer(1e-4)],
    },
}

results = core.compare(estimators, datasets, split=0.8, nfolds=10)
print(results)
