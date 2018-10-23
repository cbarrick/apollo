import os
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, cross_validate
from sklearn.externals import joblib
from dask_ml.model_selection import GridSearchCV
from distributed import Client

from apollo.prediction.Predictor import Predictor
from apollo.datasets.solar import SolarDataset
from apollo.datasets import nam

# TODO: using logging instead of printing to stdout


class SKPredictor(Predictor):

    def __init__(self, name, estimator, parameter_grid, target='UGA-C-POA-1-IRR', target_hours=tuple(np.arange(1, 25))):
        """ Predictor that works with any scikit-learn estimator

        The `sklearn.multioutput.MultiOutputRegressor` API allows this predictor to make windowed predictions with any
        estimator that has `fit` and `predict` methods.

        TODO: try to detect if the estimator supports multiple targets out-of-the-box

        Args:
            name (str):
                A descriptive human-readable name for this predictor.
                Typically the type of the regressor used such as "decision-tree" or "random forest".

            estimator (sklearn.base.BaseEstimator):
                The estimator to use for making predictions.
                The estimator must have `fit` and `predict` methods.

            parameter_grid (dict or None):
                The parameter grid to be explored during parameter tuning.

            target (str):
                The name of the target variable in the GA_POWER data.

            target_hours (Iterable[int]):
                The future hours to be predicted.
        """
        super().__init__(name, target=target, target_hours=target_hours)
        self.regressor = MultiOutputRegressor(estimator=estimator, n_jobs=1)
        self.param_grid = parameter_grid

    def save(self):
        # serialize the trained model
        path = os.path.join(self.models_dir, self.filename)
        joblib.dump(self.regressor, path)

        return path

    def load(self):
        # deserialize a saved model
        path_to_model = os.path.join(self.models_dir, self.filename)
        if os.path.exists(path_to_model):
            self.regressor = joblib.load(path_to_model)
            return self.regressor
        else:
            return None

    def train(self, start, stop, tune, num_folds):
        client = Client()  # dask scheduler
        # load dataset
        ds = SolarDataset(start=start, stop=stop, lag=1, target=self.target, target_hour=self.target_hours)
        x, y = ds.tabular()
        print('Dataset Loaded')  # TODO: write this sort of thing to a log file instead of stdout
        if tune and self.param_grid is not None:
            grid = GridSearchCV(
                estimator=self.regressor,
                param_grid=self.param_grid,
                cv=KFold(n_splits=num_folds, shuffle=True),
                scoring='neg_mean_absolute_error',
                return_train_score=False,
                n_jobs=-1,
                scheduler=client
            )
            grid.fit(x, y)
            print("Grid search completed.  Best parameters found: ")
            print(grid.best_params_)
            # save the estimator with the best parameters
            self.regressor = grid.best_estimator_
        else:
            # if a grid search is not performed, then we use default parameter values
            self.regressor.fit(x, y)

        save_location = self.save()
        return save_location

    def predict(self, reftime):
        # load the trained regressor
        self.load()
        if self.regressor is None:
            print("You must train the model before making predictions!"
                  "\nNo serialized model found at '%s'" % os.path.join(self.models_dir, self.filename))
            return None

        # load NAM data for the reftime
        previous_reftime = reftime - np.timedelta64(6, 'h')
        next_reftime = reftime + np.timedelta64(6, 'h')
        try:
            dataset = SolarDataset(start=previous_reftime, stop=next_reftime, lag=1, target=None)
        except nam.CacheMiss:
            print(f'NAM data for reftime {reftime} not cached locally.  Attempting to download it...')
            nam.open(reftime)
            dataset = SolarDataset(start=previous_reftime, stop=next_reftime, lag=1, target=None)

        data = np.asarray(dataset.tabular())[0]  # the dataset will be of length 1
        prediction = self.regressor.predict([data])[0]

        # prediction will have one predicted value for every hour in target_hours
        prediction_tuples = list()
        for idx, hour in self.target_hours:
            timestamp = np.datetime64(reftime) + np.timedelta64(hour, 'h')
            predicted_val = prediction[idx]
            prediction_tuples.append((timestamp, predicted_val))

            return prediction_tuples

    def cross_validate(self, start, stop, num_folds, metrics):
        # load hyperparams saved in training step:
        saved_model = self.load()
        if saved_model is None:
            print('WARNING: Evaluating model using default hyperparameters.  '
                  'Run `train` before calling `evaluate` to find optimal hyperparameters.')
            hyperparams = dict()
        else:
            hyperparams = saved_model.get_params()

        # load dataset
        dataset = SolarDataset(start=start, stop=stop, lag=1, target=self.target, target_hour=self.target_hours)
        x, y = dataset.tabular()
        x, y = np.asarray(x), np.asarray(y)
        print('Dataset Loaded')

        # Evaluate the classifier
        self.regressor.set_params(**hyperparams)
        scores = cross_validate(self.regressor, x, y, scoring=metrics, cv=num_folds, return_train_score=False, n_jobs=-1)

        # scores is dictionary with keys "test_<metric_name> for each metric"
        mean_scores = dict()
        for metric_name in metrics:
            mean_scores[metric_name] = np.mean(scores['test_' + metric_name])

        return mean_scores
