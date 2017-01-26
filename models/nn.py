import logging
import math

import numpy as np
import tensorflow as tf

from sklearn.base import BaseEstimator, RegressorMixin

from tensorflow.contrib.layers import *

logger = logging.getLogger(__name__)


class ConvRegressor(BaseEstimator, RegressorMixin):
    '''A 1D convnet based on MC-DNN
    http://staff.ustc.edu.cn/~cheneh/paper_pdf/2014/Yi-Zheng-WAIM2014.pdf
    '''

    def __init__(self,
            lag=96,
            activation = tf.nn.tanh,
            initializer = xavier_initializer(),
            regularizer = None,
            optimizer = tf.train.AdamOptimizer(1e-3),
            dtype = tf.float32):

        super().__init__()
        self.lag = lag
        self.activation = activation
        self.initializer = initializer
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.dtype = dtype

    def set_params(self, **params):
        super().set_params(**params)
        if hasattr(self, 'sess_'):
            del self.sess_

    def build(self, shape):
        with tf.Graph().as_default() as graph:
            self.x_ = tf.placeholder(self.dtype, [None, shape[1]])
            self.y_ = tf.placeholder(self.dtype, [None, 1])

            # Assume input is N x 1 x 96 x 3 = 288
            inputs = tf.reshape(self.x_, [-1, 1, self.lag, shape[1]//self.lag])

            #
            conv1 = convolution2d(
                inputs,
                num_outputs=8,
                kernel_size=5,
                stride=1,
                padding='SAME',
                activation_fn=self.activation,
                weights_initializer=self.initializer,
                weights_regularizer=self.regularizer,
                biases_initializer=tf.zeros_initializer)

            #
            pool1 = max_pool2d(
                conv1,
                kernel_size=[1,2],
                stride=[1,2],
                padding='SAME')

            #
            conv2 = convolution2d(
                pool1,
                num_outputs=4,
                kernel_size=5,
                stride=1,
                padding='SAME',
                activation_fn=self.activation,
                weights_initializer=self.initializer,
                weights_regularizer=self.regularizer,
                biases_initializer=tf.zeros_initializer)

            #
            pool2 = max_pool2d(
                conv2,
                kernel_size=[1,2],
                stride=[1,2],
                padding='SAME')

            #
            hidden = fully_connected(
                flatten(pool2),
                num_outputs=256,
                activation_fn=self.activation,
                weights_initializer=self.initializer,
                weights_regularizer=self.regularizer,
                biases_initializer=tf.zeros_initializer)

            # Dropout
            self.keep_prob_ = tf.placeholder(tf.float32)
            dropout = tf.nn.dropout(hidden, self.keep_prob_)

            # N x 1 = 1
            self.inference_ = fully_connected(
                dropout,
                num_outputs=1,
                activation_fn=None,
                biases_initializer=tf.zeros_initializer)

            # Loss
            loss = tf.reduce_mean(tf.squared_difference(self.inference_, self.y_))
            loss += sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            # Optimizer
            self.train_step_ = optimize_loss(
                loss,
                global_step=None,
                learning_rate=None,
                optimizer=self.optimizer)

            # Init
            init = tf.variables_initializer(tf.global_variables())
            graph.finalize()
            self.sess_ = tf.Session(graph=graph)
            self.sess_.run(init)

    def fit(self, x, y):
        return self.partial_fit(x,y)

    def partial_fit(self, x, y):
        if not hasattr(self, 'sess_'): self.build(x.shape)
        self.sess_.run(self.train_step_, {self.x_:x, self.y_:y, self.keep_prob_:0.5})
        return self

    def predict(self, x):
        return self.sess_.run(self.inference_, {self.x_:x, self.keep_prob_:1.0})


class MLPRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
            layers = (128,64,16),
            activation = tf.nn.relu,
            initializer = variance_scaling_initializer(),
            regularizer = l2_regularizer(0.01),
            optimizer = tf.train.AdamOptimizer(1e-3),
            dtype = tf.float32):

        super().__init__()
        self.layers = layers
        self.activation = activation
        self.initializer = initializer
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.dtype = dtype

    def set_params(self, **params):
        super().set_params(**params)
        if hasattr(self, 'sess_'):
            del self.sess_

    def build(self, x_shape, y_shape):
        with tf.Graph().as_default() as graph:
            # Make the batch size variable
            x_shape = (None,) + x_shape[1:]
            y_shape = (None,) + y_shape[1:]

            self.x_ = tf.placeholder(self.dtype, x_shape)
            self.y_ = tf.placeholder(self.dtype, y_shape)

            fc = self.x_
            for l in self.layers:
                fc = fully_connected(
                    fc,
                    num_outputs=l,
                    activation_fn=self.activation,
                    weights_initializer=self.initializer,
                    weights_regularizer=self.regularizer,
                    biases_initializer=tf.zeros_initializer)

            self.inference_ = fully_connected(
                fc,
                num_outputs=y_shape[-1],
                activation_fn=None,
                biases_initializer=tf.zeros_initializer)

            # Loss
            loss = tf.reduce_mean(tf.squared_difference(self.inference_, self.y_))
            loss += sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            # Optimizer
            self.train_step_ = optimize_loss(
                loss,
                global_step=None,
                learning_rate=None,
                optimizer=self.optimizer)

            # Init
            init = tf.variables_initializer(tf.global_variables())
            graph.finalize()
            self.sess_ = tf.Session(graph=graph)
            self.sess_.run(init)

    def fit(self, x, y):
        return self.partial_fit(x,y)

    def partial_fit(self, x, y):
        if not hasattr(self, 'sess_'): self.build(x.shape, y.shape)
        self.sess_.run(self.train_step_, {self.x_:x, self.y_:y})
        return self

    def predict(self, x):
        return self.sess_.run(self.inference_, {self.x_:x})
