"""
Solar Radiation Prediction with scikit's DecisionTreeRegressor
"""

import os
import json
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.externals import joblib

from apollo.datasets import simple_loader


_CACHE_DIR = "../data"  # where the NAM and GA-POWER data resides
_MODELS_DIR = "../models"  # directory where serialized models will be saved
_DEFAULT_TARGET = 'UGA-C-POA-1-IRR'


def make_model_name(target_hour, target_var):
    # creates a unique name for a model that predicts a specific target variable at a specific target hour
    return 'dtree_%shr_%s.model' % (target_hour, target_var)


def save(model, save_dir, target_hour, target_var, hyperparams={}):
    # serialize the trained model
    name = make_model_name(target_hour, target_var)
    path = os.path.join(save_dir, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    joblib.dump(model, path)

    # serialize the hyperparameters:
    hyperparams_name = name + '.hyper'
    hyperparam_path = os.path.join(save_dir, hyperparams_name)
    with open(hyperparam_path, 'w') as hyperparam_file:
        json.dump(hyperparams, hyperparam_file)

    return path


def load(save_dir, target_hour, target_var):
    # logic to load a serialized model
    name = make_model_name(target_hour, target_var)
    hyperparams_name = name + '.hyper'
    path_to_model = os.path.join(save_dir, name)
    path_to_hyperparams = os.path.join(save_dir, hyperparams_name)
    if os.path.exists(path_to_model) and os.path.exists(path_to_hyperparams):
        model = joblib.load(path_to_model)
        hyperparams = json.load(path_to_hyperparams)
        return model, hyperparams
    else:
        return None, None


def train(begin_date='2017-01-01 00:00', end_date='2017-12-31 18:00', target_hour=24, target_var=_DEFAULT_TARGET,
          cache_dir=_CACHE_DIR, save_dir=_MODELS_DIR, tune=True, num_folds=3):
    # logic to train the model using the full dataset
    X, y = simple_loader.load(start=begin_date, stop=end_date, target_hour=target_hour, target_var=target_var, cache_dir=cache_dir)
    if tune:
        print("Tuning Parameters")
        model = GridSearchCV(
            estimator=DecisionTreeRegressor(),
            param_grid={
                'splitter': ['best', 'random'],  # splitting criterion
                'max_depth': [None, 10, 20, 50, 100],  # Maximum depth of the tree. None means unbounded.
                'min_impurity_decrease': np.arange(0, 0.6, 0.05)
            },
            cv=KFold(n_splits=num_folds, shuffle=True),
            scoring='neg_mean_absolute_error',
            return_train_score=False,
            n_jobs=-1,
        )
    else:
        model = DecisionTreeRegressor()

    # train model
    model = model.fit(X, y)

    # output optimal hyperparams to the console
    if tune:
        print("Done training.  Best hyperparameters found:")
        print(model.best_params_)
        hyperparams = model.best_params_
    else:
        hyperparams = dict()

    save_location = save(model, save_dir, target_hour, target_var, hyperparams=hyperparams)
    return save_location


def evaluate(begin_date='2017-12-01 00:00', end_date='2017-12-31 18:00', target_hour=24, target_var=_DEFAULT_TARGET,
             cache_dir=_CACHE_DIR, save_dir=_MODELS_DIR, num_folds=3, metrics=['neg_mean_absolute_error']):
    # load hyperparams saved by training step:
    _, hyperparams = load(save_dir, target_hour, target_var)
    if hyperparams is None:
        print('WARNING: Evaluating model using default hyperparameters.  '
              'Run `train` before calling `evaluate` to find optimal hyperparameters.')
        hyperparams = dict()

    # Evaluate the classifier
    model = DecisionTreeRegressor(hyperparams)
    X, y = simple_loader.load(start=begin_date, stop=end_date, target_hour=target_hour, target_var=target_var, cache_dir=cache_dir)
    scores = cross_validate(model, X, y, scoring=metrics, cv=num_folds, return_train_score=False, n_jobs=-1)

    # scores is dictionary with keys "test_<metric_name> for each metric"
    mean_scores = dict()
    for metric in metrics:
        mean_scores[metric] = np.mean(scores['test_' + metric])

    return mean_scores


# TODO - need more specs from Dr. Maier
def predict(begin_date, end_date, target_hour=24, target_var=_DEFAULT_TARGET,
            cache_dir=_CACHE_DIR, save_dir=_MODELS_DIR, output_dir='../predictions'):

    model_name = make_model_name(target_hour, target_var)
    path_to_model = os.path.join(save_dir, model_name)
    model, hyperparams = load(save_dir, target_hour, target_var)
    if model is None:
        print("You must train the model before making predictions!\nNo serialized model found at '%s'" % path_to_model)
        return None
    data = simple_loader.load(start=begin_date, stop=end_date, target_var=None, cache_dir=cache_dir)[0]
    reftimes = np.arange(begin_date, end_date, dtype='datetime64[6h]')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    outpath = os.path.join(output_dir, model_name + '.out.csv')
    with open(outpath, 'w') as outfile:
        for idx, data_point in enumerate(data):
            prediction = model.predict([data_point])
            outfile.write("%s,%s\n" % (reftimes[idx], prediction[0]))

    return outpath
