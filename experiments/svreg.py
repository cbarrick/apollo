"""
Solar Radiation Prediction with scikit's Support Vector Regression
"""

import os

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.externals import joblib

from apollo.datasets import simple_loader

import numpy as np
from sklearn import svm

_CACHE_DIR = "../data"  # where the NAM and GA-POWER data resides
_MODELS_DIR = "../models"  # directory where serialized models will be saved
_DEFAULT_TARGET = 'UGA-C-POA-1-IRR'

# hyperparameters used during training, evaluation, and prediction
HYPERPARAMS = {

}


def make_model_name(target_hour, target_var):
    # creates a unique name for a model that predicts a specific target variable at a specific target hour
    return 'svreg%shr_%s.model' % (target_hour, target_var)


# TODO: export these functions to a utils module
def save(model, save_dir, target_hour, target_var):
    # logic to serialize a trained model
    name = make_model_name(target_hour, target_var)
    path = os.path.join(save_dir, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    joblib.dump(model, path)
    return path


def load(save_dir, target_hour, target_var):
    # logic to load a serialized model
    name = make_model_name(target_hour, target_var)
    path_to_model = os.path.join(save_dir, name)
    if os.path.exists(path_to_model):
        model = joblib.load(path_to_model)
        return model
    else:
        return None


def train(begin_date='2017-01-01 00:00', end_date='2017-12-31 18:00', target_hour=24, target_var=_DEFAULT_TARGET,
          cache_dir=_CACHE_DIR, save_dir=_MODELS_DIR, tune=True, num_folds=3):
    # logic to train the model using the full dataset
    X, y = simple_loader.load(start=begin_date, stop=end_date, target_hour=target_hour, target_var=target_var, cache_dir=cache_dir)
    if tune:
        model = GridSearchCV(
            estimator=svm.SVR(),
            param_grid={
                
            },
            cv=KFold(n_splits=num_folds, shuffle=True),
            scoring='mean_squared_error',
            return_train_score=False,
            n_jobs=-1,
        )
    else:
        model = svm.SVR()
    model = model.fit(X, y)
    save_location = save(model, save_dir, target_hour, target_var)


    return save_location


def evaluate(begin_date='2017-12-01 00:00', end_date='2017-12-31 18:00', target_hour=24, target_var=_DEFAULT_TARGET,
             cache_dir=_CACHE_DIR, num_folds=3):
    # logic to estimate a model's accuracy and report the results
    model = svm.SVR()
    X, y = simple_loader.load(start=begin_date, stop=end_date, target_hour=target_hour, target_var=target_var, cache_dir=cache_dir)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=num_folds, n_jobs=-1)

    return np.mean(scores)


# TODO - need more specs from Dr. Maier
def predict(begin_date, end_date, target_hour=24, target_var=_DEFAULT_TARGET,
            cache_dir=_CACHE_DIR, save_dir=_MODELS_DIR, output_dir='../predictions'):

    model_name = make_model_name(target_hour, target_var)
    path_to_model = os.path.join(save_dir, model_name)
    model = load(save_dir, target_hour, target_var)
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

