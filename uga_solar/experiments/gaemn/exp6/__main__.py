#!/usr/bin/env python3
import tensorflow as tf

from sklearn.preprocessing import scale as standard_scale

from uga_solar.data import gaemn
from .. import core

from .models import ConvRegressor, MLPRegressor


core.setup()

datasets = {
    gaemn.GaemnLoader: {
        'path'       : ['./gaemn.zip'],
        'years'      : [range(2003,2013)],
        'x_features' : [('day', 'time', 'solar radiation')],
        'y_features' : [('solar radiation (+96)',)],
        'lag'        : [96],
        'scale'      : [standard_scale],
    },
} # yapf: disable

estimators = {
    ConvRegressor: [{
        'lag': [96],
        'activation': [tf.nn.elu, tf.nn.tanh],
        'initializer': [
            tf.contrib.layers.xavier_initializer(),
            tf.contrib.layers.variance_scaling_initializer(2)
        ],
        'regularizer': [tf.contrib.layers.l2_regularizer(0.01), None],
        'optimizer': [tf.train.AdamOptimizer(1e-4)],
    }],
    MLPRegressor: {
        'layers': [(32, ), (64, ), (128, ), (64, 32)],
        'activation': [tf.nn.elu, tf.nn.tanh],
        'initializer': [
            tf.contrib.layers.xavier_initializer(),
            tf.contrib.layers.variance_scaling_initializer(2)
        ],
        'regularizer': [tf.contrib.layers.l2_regularizer(0.01), None],
        'optimizer': [tf.train.AdamOptimizer(1e-4)],
    },
}

results = core.percent_split(estimators, datasets, 0.8, nfolds=10)
print(results)
