#!/usr/bin/env python3
''' Experiment 12

This experiment tests the neural net models on 24hr predictions using 24hr lag.
Many combinations of hyper parameters were compared. The results inicate that
MLP networks performe significantly better than the MC-DNN (1D convoltional)
networks. The results also indicate that elu activation performs significantly
better than tanh activation.

Comparing initializers is fruitless here because xavier_initializer is a wrapper
around variance_scaling_initializer, and the output of this experiemnt only
distinguishes the initializers by memory location without giving the parameters.

The top two are not significantly different in results, but differ both in
initializer and regularizer. Perhaps the architecture and activation are more
important than the initializer and regularizer.

Results:
```
METRIC  TRIAL
------------------------------------------------------------------------
64.904  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(64, 32), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

65.037  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(64, 32), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

65.514  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(64, 32), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

66.900  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(32,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

66.916  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(128,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

67.008  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(128,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

67.347  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(128,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

67.372  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(128,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

67.479  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(64,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

67.593  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(32,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

67.773  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(64, 32), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

67.923  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(64,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

67.988  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(32,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

67.998  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(64, 32), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

68.048  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(128,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

68.187  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(64,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

68.245  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(128,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

68.270  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(128,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

68.313  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(128,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

68.358  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(64, 32), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

68.415  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(64, 32), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

68.505  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(64,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

68.546  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(64,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

68.686  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(64,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

68.919  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(64,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

69.087  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(32,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

69.130  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(32,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

69.216  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(64,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

69.312  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(32,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93bae8>)

69.521  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bc80>, layers=(32,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

69.754  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(64, 32), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

69.903  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        ConvRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bf28>, lag=96, optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f416f60>, regularizer=None)

69.956  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        MLPRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bbf8>, layers=(32,), optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f3f3470>, regularizer=None)

70.631  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        ConvRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bf28>, lag=96, optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f416f60>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93be18>)

71.018  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        ConvRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10032cf28>, lag=96, optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f416f60>, regularizer=None)

73.192  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        ConvRegressor(activation=<function elu at 0x107ae8b70>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10032cf28>, lag=96, optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f416f60>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93be18>)

73.615  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        ConvRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bf28>, lag=96, optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f416f60>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93be18>)

95.545  DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        ConvRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10032cf28>, lag=96, optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f416f60>, regularizer=None)

101.814 DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        ConvRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10f93bf28>, lag=96, optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f416f60>, regularizer=None)

110.229 DataSet(path='./gaemn15.zip', city='GRIFFIN', years=(2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012), x_features=('day', 'time', 'solar radiation'), y_features=('solar radiation (+96)',), lag=96)
        ConvRegressor(activation=<function tanh at 0x10795c7b8>, dtype=tf.float32, initializer=<function variance_scaling_initializer.<locals>._initializer at 0x10032cf28>, lag=96, optimizer=<tensorflow.python.training.adam.AdamOptimizer object at 0x10f416f60>, regularizer=<function l2_regularizer.<locals>.l2 at 0x10f93be18>)


t-Test Matrix (p-values)
------------------------------------------------------------------------
   --     52.515%   1.129%   0.027%   0.083%   0.006%   0.053%   0.045%   0.001%   0.055%   0.195%   0.052%   0.028%   0.065%   0.001%   0.101%   0.000%   0.003%   0.000%   0.144%   0.045%   0.000%   0.001%   0.005%   0.002%   0.005%   0.008%   0.054%   0.002%   0.055%   0.010%   0.043%   0.003%   0.000%   0.001%   0.000%   0.000%   0.001%   0.000%   0.000%
 52.515%    --      1.266%   0.004%   0.011%   0.000%   0.014%   0.014%   0.000%   0.056%   0.057%   0.037%   0.009%   0.019%   0.000%   0.113%   0.000%   0.000%   0.000%   0.046%   0.010%   0.000%   0.000%   0.000%   0.000%   0.000%   0.006%   0.041%   0.000%   0.014%   0.003%   0.017%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%
  1.129%   1.266%    --      0.268%   0.652%   0.028%   0.310%   0.367%   0.011%   0.653%   0.195%   0.385%   0.160%   0.071%   0.003%   0.664%   0.000%   0.003%   0.002%   0.155%   0.045%   0.000%   0.001%   0.003%   0.003%   0.004%   0.056%   0.214%   0.001%   0.040%   0.017%   0.058%   0.002%   0.000%   0.001%   0.000%   0.000%   0.001%   0.000%   0.000%
  0.027%   0.004%   0.268%    --     89.910%  37.875%   1.234%   1.416%   0.124%   4.852%  20.722%   1.096%   0.268%   7.538%   0.358%   2.248%   0.077%   0.239%   0.109%   5.953%   2.109%   0.004%   0.003%   0.239%   0.040%   0.014%   0.030%   0.241%   0.058%   0.768%   0.039%   0.449%   0.012%   0.000%   0.001%   0.000%   0.000%   0.001%   0.000%   0.000%
  0.083%   0.011%   0.652%  89.910%    --     63.623%   1.426%   0.379%   2.356%   4.269%  20.959%   0.548%   0.024%   7.036%   0.190%   2.064%   0.066%   0.116%   0.049%   5.087%   1.503%   0.002%   0.001%   0.140%   0.009%   0.005%   0.014%   0.135%   0.047%   0.648%   0.015%   0.343%   0.007%   0.000%   0.000%   0.000%   0.000%   0.001%   0.000%   0.000%
  0.006%   0.000%   0.028%  37.875%  63.623%    --     13.972%  16.918%   0.432%  16.926%  23.754%   5.489%   2.213%   9.944%   0.758%   6.108%   0.069%   0.338%   0.210%   7.317%   2.845%   0.004%   0.006%   0.219%   0.070%   0.023%   0.143%   0.727%   0.036%   0.724%   0.087%   0.549%   0.011%   0.000%   0.003%   0.000%   0.000%   0.001%   0.000%   0.000%
  0.053%   0.014%   0.310%   1.234%   1.426%  13.972%    --     80.742%  54.582%  36.891%  57.570%   5.426%   2.784%  33.179%  10.532%   5.693%   4.058%   5.132%   3.133%  20.743%  11.405%   0.414%   0.205%   2.490%   0.515%   0.121%   0.025%   0.233%   0.524%   2.485%   0.162%   1.451%   0.059%   0.003%   0.003%   0.001%   0.000%   0.001%   0.000%   0.000%
  0.045%   0.014%   0.367%   1.416%   0.379%  16.918%  80.742%    --     63.915%  27.997%  59.977%   1.614%   0.504%  34.084%   9.239%   4.282%   4.081%   4.735%   2.212%  21.513%  11.488%   0.325%   0.214%   2.714%   0.337%   0.160%   0.006%   0.149%   0.680%   2.989%   0.109%   1.371%   0.088%   0.003%   0.001%   0.001%   0.000%   0.001%   0.000%   0.000%
  0.001%   0.000%   0.011%   0.124%   2.356%   0.432%  54.582%  63.915%    --     73.156%  67.862%  25.300%  17.536%  40.312%  11.474%  17.453%   3.457%   6.621%   2.114%  26.155%  16.157%   0.193%   0.443%   3.313%   0.746%   0.541%   0.315%   1.664%   0.624%   3.576%   0.486%   2.049%   0.162%   0.005%   0.010%   0.001%   0.000%   0.001%   0.000%   0.000%
  0.055%   0.056%   0.653%   4.852%   4.269%  16.926%  36.891%  27.997%  73.156%    --     84.464%   2.690%  15.804%  60.697%  36.854%   1.818%  23.355%  24.174%  13.246%  41.256%  29.910%   6.085%   5.949%  13.369%   3.283%   2.647%   0.001%   0.100%   4.180%   9.087%   0.767%   4.222%   0.917%   0.050%   0.008%   0.007%   0.000%   0.002%   0.000%   0.000%
  0.195%   0.057%   0.195%  20.722%  20.959%  23.754%  57.570%  59.977%  67.862%  84.464%    --     86.869%  77.404%  37.832%  57.712%  70.921%  28.374%  25.119%  37.717%   2.936%   6.931%  19.093%  13.919%   1.266%   3.401%   1.463%  19.618%  22.325%   0.033%   0.129%   0.882%   0.527%   0.028%   0.011%   0.101%   0.000%   0.000%   0.000%   0.000%   0.000%
  0.052%   0.037%   0.385%   1.096%   0.548%   5.489%   5.426%   1.614%  25.300%   2.690%  86.869%    --     74.988%  92.102%  79.271%  24.516%  53.881%  51.288%  37.902%  62.186%  50.068%  20.236%  17.104%  26.092%   7.202%   5.145%   0.004%   0.131%   8.510%  13.764%   0.947%   6.030%   1.482%   0.073%   0.005%   0.008%   0.000%   0.001%   0.000%   0.000%
  0.028%   0.009%   0.160%   0.268%   0.024%   2.213%   2.784%   0.504%  17.536%  15.804%  77.404%  74.988%    --     98.739%  86.477%  63.104%  51.963%  46.078%  36.795%  60.722%  44.949%  13.876%   9.405%  18.362%   2.192%   1.545%   0.292%   1.406%   4.391%   9.205%   0.133%   2.920%   0.446%   0.005%   0.000%   0.001%   0.000%   0.001%   0.000%   0.000%
  0.065%   0.019%   0.071%   7.538%   7.036%   9.944%  33.179%  34.084%  40.312%  60.697%  37.832%  92.102%  98.739%    --     88.530%  84.365%  48.275%  38.516%  50.206%   9.862%   9.741%  26.889%  17.772%   3.567%   2.428%   1.753%  21.250%  23.424%   0.519%   1.545%   0.700%   1.140%   0.105%   0.009%   0.048%   0.000%   0.000%   0.000%   0.000%   0.000%
  0.001%   0.000%   0.003%   0.358%   0.190%   0.758%  10.532%   9.239%  11.474%  36.854%  57.712%  79.271%  86.477%  88.530%    --     83.738%  16.742%  11.745%   6.634%  53.105%  32.740%   1.707%   1.805%   3.617%   0.046%   0.885%   8.746%  13.218%   0.872%   5.277%   0.370%   2.174%   0.158%   0.001%   0.004%   0.000%   0.000%   0.001%   0.000%   0.000%
  0.101%   0.113%   0.664%   2.248%   2.064%   6.108%   5.693%   4.282%  17.453%   1.818%  70.921%  24.516%  63.104%  84.365%  83.738%    --     93.421%  90.856%  83.764%  87.410%  80.678%  60.858%  56.767%  56.239%  31.740%  23.488%   0.056%   0.138%  24.211%  28.148%   6.751%  15.783%   6.907%   0.881%   0.139%   0.059%   0.001%   0.002%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.077%   0.066%   0.069%   4.058%   4.081%   3.457%  23.355%  28.374%  53.881%  51.963%  48.275%  16.742%  93.421%    --     84.203%  74.920%  81.311%  63.845%   6.885%   5.128%   5.536%   0.883%   1.173%  17.074%  22.793%   0.461%   6.297%   0.919%   3.342%   0.098%   0.001%   0.012%   0.000%   0.000%   0.001%   0.000%   0.000%
  0.003%   0.000%   0.003%   0.239%   0.116%   0.338%   5.132%   4.735%   6.621%  24.174%  25.119%  51.288%  46.078%  38.516%  11.745%  90.856%  84.203%    --     84.954%  83.593%  59.816%  24.422%  10.419%   3.906%   0.077%   0.639%  19.031%  23.396%   0.783%   5.372%   0.282%   2.011%   0.063%   0.000%   0.003%   0.000%   0.000%   0.001%   0.000%   0.000%
  0.000%   0.000%   0.002%   0.109%   0.049%   0.210%   3.133%   2.212%   2.114%  13.246%  37.717%  37.902%  36.795%  50.206%   6.634%  83.764%  74.920%  84.954%    --     93.986%  82.898%  28.163%  33.967%  31.002%   2.354%   8.114%  14.650%  20.601%   6.077%  14.285%   1.845%   6.254%   1.059%   0.014%   0.012%   0.003%   0.000%   0.001%   0.000%   0.000%
  0.144%   0.046%   0.155%   5.953%   5.087%   7.317%  20.743%  21.513%  26.155%  41.256%   2.936%  62.186%  60.722%   9.862%  53.105%  87.410%  81.311%  83.593%  93.986%    --     83.095%  79.535%  70.767%  33.424%  22.130%  11.667%  43.968%  43.400%   3.281%   1.539%   2.815%   2.066%   0.286%   0.091%   0.243%   0.001%   0.000%   0.000%   0.000%   0.000%
  0.045%   0.010%   0.045%   2.109%   1.503%   2.845%  11.405%  11.488%  16.157%  29.910%   6.931%  50.068%  44.949%   9.741%  32.740%  80.678%  63.845%  59.816%  82.898%  83.095%    --     84.273%  72.401%  30.507%  14.222%   4.855%  39.901%  39.998%   2.815%   4.088%   0.578%   1.764%   0.109%   0.008%   0.035%   0.000%   0.000%   0.000%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.004%   0.002%   0.004%   0.414%   0.325%   0.193%   6.085%  19.093%  20.236%  13.876%  26.889%   1.707%  60.858%   6.885%  24.422%  28.163%  79.535%  84.273%    --     77.487%  52.883%   7.519%   8.740%  24.022%  31.026%   5.036%  16.026%   2.554%   7.461%   0.599%   0.003%   0.016%   0.001%   0.000%   0.001%   0.000%   0.000%
  0.001%   0.000%   0.001%   0.003%   0.001%   0.006%   0.205%   0.214%   0.443%   5.949%  13.919%  17.104%   9.405%  17.772%   1.805%  56.767%   5.128%  10.419%  33.967%  70.767%  72.401%  77.487%    --     56.756%   9.414%   1.845%  28.383%  32.741%   4.211%  14.287%   1.669%   7.410%   0.178%   0.002%   0.013%   0.001%   0.000%   0.001%   0.000%   0.000%
  0.005%   0.000%   0.003%   0.239%   0.140%   0.219%   2.490%   2.714%   3.313%  13.369%   1.266%  26.092%  18.362%   3.567%   3.617%  56.239%   5.536%   3.906%  31.002%  33.424%  30.507%  52.883%  56.756%    --     30.049%  10.555%  55.251%  54.773%   0.604%   7.871%   3.186%   5.301%   0.032%   0.003%   0.072%   0.000%   0.000%   0.001%   0.000%   0.000%
  0.002%   0.000%   0.003%   0.040%   0.009%   0.070%   0.515%   0.337%   0.746%   3.283%   3.401%   7.202%   2.192%   2.428%   0.046%  31.740%   0.883%   0.077%   2.354%  22.130%  14.222%   7.519%   9.414%  30.049%    --     54.589%  72.233%  68.198%  29.596%  33.311%   3.782%  11.827%   1.952%   0.007%   0.010%   0.001%   0.000%   0.001%   0.000%   0.000%
  0.005%   0.000%   0.004%   0.014%   0.005%   0.023%   0.121%   0.160%   0.541%   2.647%   1.463%   5.145%   1.545%   1.753%   0.885%  23.488%   1.173%   0.639%   8.114%  11.667%   4.855%   8.740%   1.845%  10.555%  54.589%    --     94.444%  85.860%  45.070%  38.973%   7.557%  19.086%   0.072%   0.009%   0.081%   0.001%   0.000%   0.001%   0.000%   0.000%
  0.008%   0.006%   0.056%   0.030%   0.014%   0.143%   0.025%   0.006%   0.315%   0.001%  19.618%   0.004%   0.292%  21.250%   8.746%   0.056%  17.074%  19.031%  14.650%  43.968%  39.901%  24.022%  28.383%  55.251%  72.233%  94.444%    --     71.453%  82.130%  71.608%  35.380%  44.976%  29.689%   3.742%   0.478%   0.100%   0.001%   0.002%   0.000%   0.000%
  0.054%   0.041%   0.214%   0.241%   0.135%   0.727%   0.233%   0.149%   1.664%   0.100%  22.325%   0.131%   1.406%  23.424%  13.218%   0.138%  22.793%  23.396%  20.601%  43.400%  39.998%  31.026%  32.741%  54.773%  68.198%  85.860%  71.453%    --     91.948%  79.382%  46.701%  53.559%  40.435%   9.155%   1.486%   0.289%   0.008%   0.003%   0.000%   0.000%
  0.002%   0.000%   0.001%   0.058%   0.047%   0.036%   0.524%   0.680%   0.624%   4.180%   0.033%   8.510%   4.391%   0.519%   0.872%  24.211%   0.461%   0.783%   6.077%   3.281%   2.815%   5.036%   4.211%   0.604%  29.596%  45.070%  82.130%  91.948%    --     58.787%  41.323%  33.800%   1.662%   0.219%   1.575%   0.001%   0.000%   0.001%   0.000%   0.000%
  0.055%   0.014%   0.040%   0.768%   0.648%   0.724%   2.485%   2.989%   3.576%   9.087%   0.129%  13.764%   9.205%   1.545%   5.277%  28.148%   6.297%   5.372%  14.285%   1.539%   4.088%  16.026%  14.287%   7.871%  33.311%  38.973%  71.608%  79.382%  58.787%    --     71.238%  45.348%  21.995%   6.189%   7.640%   0.007%   0.005%   0.000%   0.000%   0.000%
  0.010%   0.003%   0.017%   0.039%   0.015%   0.087%   0.162%   0.109%   0.486%   0.767%   0.882%   0.947%   0.133%   0.700%   0.370%   6.751%   0.919%   0.282%   1.845%   2.815%   0.578%   2.554%   1.669%   3.186%   3.782%   7.557%  35.380%  46.701%  41.323%  71.238%    --     72.946%  63.560%   1.082%   0.010%   0.001%   0.000%   0.001%   0.000%   0.000%
  0.043%   0.017%   0.058%   0.449%   0.343%   0.549%   1.451%   1.371%   2.049%   4.222%   0.527%   6.030%   2.920%   1.140%   2.174%  15.783%   3.342%   2.011%   6.254%   2.066%   1.764%   7.461%   7.410%   5.301%  11.827%  19.086%  44.976%  53.559%  33.800%  45.348%  72.946%    --     92.243%  14.506%   7.282%   0.001%   0.013%   0.000%   0.000%   0.000%
  0.003%   0.000%   0.002%   0.012%   0.007%   0.011%   0.059%   0.088%   0.162%   0.917%   0.028%   1.482%   0.446%   0.105%   0.158%   6.907%   0.098%   0.063%   1.059%   0.286%   0.109%   0.599%   0.178%   0.032%   1.952%   0.072%  29.689%  40.435%   1.662%  21.995%  63.560%  92.243%    --      4.588%   6.477%   0.004%   0.000%   0.001%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.003%   0.003%   0.005%   0.050%   0.011%   0.073%   0.005%   0.009%   0.001%   0.881%   0.001%   0.000%   0.014%   0.091%   0.008%   0.003%   0.002%   0.003%   0.007%   0.009%   3.742%   9.155%   0.219%   6.189%   1.082%  14.506%   4.588%    --     24.498%   0.001%   0.000%   0.001%   0.000%   0.000%
  0.001%   0.000%   0.001%   0.001%   0.000%   0.003%   0.003%   0.001%   0.010%   0.008%   0.101%   0.005%   0.000%   0.048%   0.004%   0.139%   0.012%   0.003%   0.012%   0.243%   0.035%   0.016%   0.013%   0.072%   0.010%   0.081%   0.478%   1.486%   1.575%   7.640%   0.010%   7.282%   6.477%  24.498%    --      0.118%   0.006%   0.001%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.001%   0.001%   0.001%   0.007%   0.000%   0.008%   0.001%   0.000%   0.000%   0.059%   0.000%   0.000%   0.003%   0.001%   0.000%   0.001%   0.001%   0.000%   0.001%   0.001%   0.100%   0.289%   0.001%   0.007%   0.001%   0.001%   0.004%   0.001%   0.118%    --     36.433%   0.002%   0.000%   0.000%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.001%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.001%   0.008%   0.000%   0.005%   0.000%   0.013%   0.000%   0.000%   0.006%  36.433%    --      0.004%   0.000%   0.000%
  0.001%   0.000%   0.001%   0.001%   0.001%   0.001%   0.001%   0.001%   0.001%   0.002%   0.000%   0.001%   0.001%   0.000%   0.001%   0.002%   0.001%   0.001%   0.001%   0.000%   0.000%   0.001%   0.001%   0.001%   0.001%   0.001%   0.002%   0.003%   0.001%   0.000%   0.001%   0.000%   0.001%   0.001%   0.001%   0.002%   0.004%    --      2.773%   0.883%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   2.773%    --      1.609%
  0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.000%   0.883%   1.609%    --
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
