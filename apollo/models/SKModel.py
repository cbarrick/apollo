""" Generic Model that works with any scikit-learn estimator.
"""

import os
import numpy as np
from apollo.datasets.solar import SolarDataset
from apollo.models.base import Model
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.externals import joblib

# scoring metrics
_DEFAULT_METRICS = {
    'mae': make_scorer(mean_absolute_error),
    'mse': make_scorer(mean_squared_error),
    'r2': make_scorer(r2_score)
}


class SKModel(Model):

    def __init__(self, name, regressor, parameter_grid):
        """ Initializes a Model that wraps a scikit-learn regressor

        Args:
            name (str):
                human-readable name for the model.
            regressor (class):
                class of the scikit-learn regressor to use.
                This should be the regressor class itself, not an instance of a regressor.
            parameter_grid (dict):
                parameter grid explored during parameter tuning
        """
        super().__init__(name)
        self.regressor = regressor
        self.param_grid = parameter_grid

    def save(self, model, save_dir, target_hour, target_var):
        # serialize the trained model
        name = self._generate_name(target_hour, target_var)
        path = os.path.join(save_dir, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        joblib.dump(model, path)

        return path

    def load(self, save_dir, target_hour, target_var):
        # deserialize a saved model
        name = self._generate_name(target_hour, target_var)
        path_to_model = os.path.join(save_dir, name)
        if os.path.exists(path_to_model):
            model = joblib.load(path_to_model)
            return model
        else:
            return None

    def train(self, begin_date, end_date, target_hour, target_var, cache_dir, save_dir, tune=True, num_folds=3):
        dataset = SolarDataset(start=begin_date, stop=end_date,
                               target=target_var, target_hour=target_hour,
                               cache_dir=cache_dir)
        # convert dataset to tabular form accepted by scikit estimators
        x, y = dataset.tabular()
        print('Dataset Loaded')
        if tune and self.param_grid is not None:
            grid = GridSearchCV(
                estimator=self.regressor(),
                param_grid=self.param_grid,
                cv=KFold(n_splits=num_folds, shuffle=True),
                scoring='neg_mean_absolute_error',
                return_train_score=False,
                n_jobs=-1,
            )
            grid.fit(x, y)
            print("Grid search completed.  Best parameters found: ")
            print(grid.best_params_)
            # save the estimator with the best parameters
            model = grid.best_estimator_
        else:
            # if a grid search is not performed, then we use default parameter values
            model = self.regressor()
            model = model.fit(x, y)

        save_location = self.save(model, save_dir, target_hour, target_var)
        return save_location

    def evaluate(self, begin_date, end_date, target_hour, target_var, cache_dir, save_dir,
                 num_folds=3, metrics=_DEFAULT_METRICS):
        # load hyperparams saved in training step:
        saved_model = self.load(save_dir, target_hour, target_var)
        if saved_model is None:
            print('WARNING: Evaluating model using default hyperparameters.  '
                  'Run `train` before calling `evaluate` to find optimal hyperparameters.')
            hyperparams = dict()
        else:
            hyperparams = saved_model.get_params()

        # load dataset
        dataset = SolarDataset(start=begin_date, stop=end_date,
                               target=target_var, target_hour=target_hour,
                               cache_dir=cache_dir)
        x, y = dataset.tabular()
        print('Dataset Loaded')

        # Evaluate the classifier
        model = self.regressor(**hyperparams)
        scores = cross_validate(model, x, y, scoring=metrics, cv=num_folds, return_train_score=False, n_jobs=-1)

        # scores is dictionary with keys "test_<metric_name> for each metric"
        mean_scores = dict()
        for metric_name in metrics:
            mean_scores[metric_name] = np.mean(scores['test_' + metric_name])

        return mean_scores

    def predict(self, begin_date, end_date, target_hour, target_var, cache_dir, save_dir):
        # load the trained model
        model_name = self._generate_name(target_hour, target_var)
        path_to_model = os.path.join(save_dir, model_name)
        model = self.load(save_dir, target_hour, target_var)
        if model is None:
            print("You must train the model before making predictions!"
                  "\nNo serialized model found at '%s'" % path_to_model)
            return None

        # load NAM data without labels
        dataset = SolarDataset(start=begin_date, stop=end_date, target=None, cache_dir=cache_dir)
        data = dataset.tabular()
        print('Dataset Loaded')
        reftimes = np.arange(begin_date, end_date, dtype='datetime64[6h]')

        predictions = []
        # make predictions
        for idx, data_point in enumerate(data):
            prediction = model.predict([data_point])
            timestamp = reftimes[idx]
            data_point = [timestamp, prediction[0]]
            predictions.append(data_point)

        return predictions
