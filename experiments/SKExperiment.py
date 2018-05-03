"""
SKExperiment represents a scikit-learn experiment.

The constructor allows experiments to be created from any scikit-learn regression model
"""

import os
import numpy as np
from apollo.datasets import simple_loader
from experiments.Experiment import Experiment
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.externals import joblib


_CACHE_DIR = '../data'                  # where the NAM and GA-POWER data resides
_MODELS_DIR = '../models'               # directory where serialized models will be saved
_OUTPUT_DIR = '../predictions'          # directory where predictions are saved
_DEFAULT_TARGET = 'UGA-C-POA-1-IRR'     # name of target var

# scoring metrics
_DEFAULT_METRICS = {
    'mae': make_scorer(mean_absolute_error),
    'r2': make_scorer(r2_score)
}


class SKExperiment(Experiment):

    def __init__(self, name, regressor, parameter_grid, metrics=_DEFAULT_METRICS):
        super().__init__(name)
        self.regressor = regressor
        self.param_grid = parameter_grid
        self.metrics = metrics

    def save(self, model, save_dir, target_hour, target_var):
        # serialize the trained model
        name = self.make_model_name(target_hour, target_var)
        path = os.path.join(save_dir, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        joblib.dump(model, path)

        return path

    def load(self, save_dir, target_hour, target_var):
        # deserialize a saved model
        name = self.make_model_name(target_hour, target_var)
        path_to_model = os.path.join(save_dir, name)
        if os.path.exists(path_to_model):
            model = joblib.load(path_to_model)
            return model
        else:
            return None

    def train(self, begin_date='2017-12-01 00:00', end_date='2017-12-31 18:00', target_hour=24, target_var=_DEFAULT_TARGET,
              cache_dir=_CACHE_DIR, save_dir=_MODELS_DIR, tune=True, num_folds=3):
        # trains the model using the dataset between the begin and end date
        X, y = simple_loader.load(start=begin_date, stop=end_date, target_hour=target_hour, target_var=target_var, cache_dir=cache_dir)
        if tune and self.param_grid is not None:
            grid = GridSearchCV(
                estimator=self.regressor(),
                param_grid=self.param_grid,
                cv=KFold(n_splits=num_folds, shuffle=True),
                scoring='neg_mean_absolute_error',
                return_train_score=False,
                n_jobs=-1,
            )
            grid.fit(X, y)
            print("Grid search completed.  Best parameters found: ")
            print(grid.best_params_)
            model = grid.best_estimator_
        else:
            model = self.regressor()
            model = model.fit(X, y)

        save_location = self.save(model, save_dir, target_hour, target_var)
        return save_location

    def evaluate(self, begin_date='2017-12-01 00:00', end_date='2017-12-31 18:00', target_hour=24,
                 target_var=_DEFAULT_TARGET, cache_dir=_CACHE_DIR, save_dir=_MODELS_DIR, num_folds=3):
        # load hyperparams saved in training step:
        saved_model = self.load(save_dir, target_hour, target_var)
        if saved_model is None:
            print('WARNING: Evaluating model using default hyperparameters.  '
                  'Run `train` before calling `evaluate` to find optimal hyperparameters.')
            hyperparams = dict()
        else:
            hyperparams = saved_model.get_params()

        # Evaluate the classifier
        model = self.regressor(**hyperparams)
        X, y = simple_loader.load(start=begin_date, stop=end_date, target_hour=target_hour, target_var=target_var,
                                  cache_dir=cache_dir)
        scores = cross_validate(model, X, y, scoring=self.metrics, cv=num_folds, return_train_score=False, n_jobs=-1)

        # scores is dictionary with keys "test_<metric_name> for each metric"
        mean_scores = dict()
        for metric in self.metrics:
            mean_scores[metric] = np.mean(scores['test_' + metric])

        return mean_scores

    # TODO: output predictions in format required by Dr. Maier
    def predict(self, begin_date, end_date, target_hour=24, target_var=_DEFAULT_TARGET,
                cache_dir=_CACHE_DIR, save_dir=_MODELS_DIR, output_dir=_OUTPUT_DIR):
        model_name = self.make_model_name(target_hour, target_var)
        path_to_model = os.path.join(save_dir, model_name)
        model = self.load(save_dir, target_hour, target_var)
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
