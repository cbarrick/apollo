import logging
import numpy as np
import os

from dask_ml.model_selection import GridSearchCV
from distributed import Client

from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from apollo.prediction.Predictor import Predictor
from apollo.datasets.solar import SolarDataset
from apollo.datasets import nam

logger = logging.getLogger(__name__)


class SKPredictor(Predictor):

    @classmethod
    def get_name(cls):
        return 'skpredictor'

    def __init__(self, estimator, parameter_grid, default_params, target='UGA-C-POA-1-IRR', target_hours=tuple(np.arange(1, 25))):
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

            good_params (dict or None):
                Hyperparameters that should be used as a default

            target (str):
                The name of the target variable in the GA_POWER data.

            target_hours (Iterable[int]):
                The future hours to be predicted.
        """
        super().__init__(target=target, target_hours=target_hours)
        self.regressor = MultiOutputRegressor(estimator=estimator, n_jobs=1)
        self.param_grid = parameter_grid
        self.default_params = default_params

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
        logger.debug('Dataset Loaded')
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
            logger.info(f'Grid search completed.  Best parameters found: \n{grid.best_params_}')
            # save the estimator with the best parameters
            self.regressor = grid.best_estimator_
        else:
            # if a grid search is not performed, then we use default parameter values
            self.regressor.fit(x, y)

        save_location = self.save()
        return save_location

    def cross_validate(self, start, stop, num_folds, metrics):
        # load hyperparams saved in training step:
        saved_model = self.load()
        if saved_model is None:
            logger.warning(f'Evaluating model using default hyperparameters. '
                           f'Run `train` before calling `evaluate` to find optimal hyperparameters.')
            hyperparams = dict()
        else:
            hyperparams = saved_model.get_params()

        # load dataset
        dataset = SolarDataset(start=start, stop=stop, lag=1, target=self.target, target_hour=self.target_hours)
        x, y = dataset.tabular()
        x, y = np.asarray(x), np.asarray(y)
        logger.debug('Dataset Loaded')

        # Evaluate the classifier
        self.regressor.set_params(**hyperparams)
        scores = cross_validate(self.regressor, x, y, scoring=metrics, cv=num_folds, return_train_score=False, n_jobs=-1)

        # scores is dictionary with keys "test_<metric_name> for each metric"
        mean_scores = dict()
        for metric_name in metrics:
            mean_scores[metric_name] = np.mean(scores['test_' + metric_name])

        return mean_scores

    def predict(self, reftime):
        # load the trained regressor
        self.load()
        if self.regressor is None:
            logger.error(f'You must train the model before making predictions!\n'
                         f'No serialize model found at {os.path.join(self.models_dir, self.filename)}')
            return None

        # get small window around reftime (since lag is nonzero)
        previous_reftime = np.datetime64(reftime) - np.timedelta64(6, 'h')
        next_reftime = np.datetime64(reftime) + np.timedelta64(6, 'h')
        # ensure NAM data is cached localled before making prediction
        nam.open(previous_reftime, reftime)
        dataset = SolarDataset(start=previous_reftime, stop=next_reftime, lag=1, target=None)

        data = np.asarray(dataset.tabular())[0]  # the dataset will be of length 1
        prediction = self.regressor.predict([data])[0]

        # prediction will have one predicted value for every hour in target_hours
        prediction_tuples = list()
        for idx, hour in enumerate(self.target_hours):
            timestamp = np.datetime64(reftime) + np.timedelta64(int(hour), 'h')
            predicted_val = prediction[idx]
            prediction_tuples.append((timestamp, predicted_val))

        return prediction_tuples


# define scikit predictors

class LinearRegressionPredictor(SKPredictor):
    @classmethod
    def get_name(cls):
        return 'linreg'

    def __init__(self, target='UGA-C-POA-1-IRR', target_hours=tuple(np.arange(1, 25))):
        super().__init__(
            estimator=LinearRegression(),
            parameter_grid=None,
            default_params=None,
            target=target,
            target_hours=target_hours)


class KNearestPredictor(SKPredictor):
    @classmethod
    def get_name(cls):
        return 'knn'

    def __init__(self, target='UGA-C-POA-1-IRR', target_hours=tuple(np.arange(1, 25))):
        super().__init__(
            estimator=KNeighborsRegressor(),
            parameter_grid={
                'estimator__n_neighbors': np.arange(3, 15, 2),             # k
                'estimator__weights': ['uniform', 'distance'],             # how are neighboring values weighted
            },
            default_params={
                'estimator__n_neighbors': 5,
                'estimator__weights': 'distance',
            },
            target=target,
            target_hours=target_hours)


class SupportVectorPredictor(SKPredictor):
    @classmethod
    def get_name(cls):
        return 'svr'

    def __init__(self, target='UGA-C-POA-1-IRR', target_hours=tuple(np.arange(1, 25))):
        super().__init__(
            estimator=SVR(),
            parameter_grid={
                'estimator__C': np.arange(1.0, 1.6, 0.2),                  # penalty parameter C of the error term
                'estimator__epsilon': np.arange(0.4, 0.8, 0.1),            # width of the no-penalty region
                'estimator__kernel': ['rbf', 'sigmoid'],                   # kernel function
                'estimator__gamma': [0.001, 0.0025, 0.005]                 # kernel coefficient
            },
            default_params={
                'estimator__C': 1.4,
                'estimator__epsilon': 0.6,
                'estimator__kernel': 'sigmoid',
                'estimator__gamma': 0.001
            },
            target=target,
            target_hours=target_hours)


class DTreePredictor(SKPredictor):
    @classmethod
    def get_name(cls):
        return 'dtree'

    def __init__(self, target='UGA-C-POA-1-IRR', target_hours=tuple(np.arange(1, 25))):
        super().__init__(
            estimator=DecisionTreeRegressor(),
            parameter_grid={
                'estimator__splitter': ['best', 'random'],         # splitting criterion
                'estimator__max_depth': [None, 10, 20, 30],        # Maximum depth of the tree. None means unbounded.
                'estimator__min_impurity_decrease': np.arange(0.15, 0.40, 0.05)
            },
            default_params={
                'estimator__splitter': 'best',
                'estimator__max_depth': 20,
                'estimator__min_impurity_decrease': 0.25
            },
            target=target,
            target_hours=target_hours)


class RandomForestPredictor(SKPredictor):
    @classmethod
    def get_name(cls):
        return 'rf'

    def __init__(self, target='UGA-C-POA-1-IRR', target_hours=tuple(np.arange(1, 25))):
        super().__init__(
            estimator=RandomForestRegressor(),
            parameter_grid={
                'estimator__n_estimators': [10, 50, 100, 250],
                'estimator__max_depth': [None, 10, 20, 30],
                'estimator__min_impurity_decrease': np.arange(0.15, 0.40, 0.05)
            },
            default_params={
                'estimator__n_estimators': 100,
                'estimator__max_depth': 50,
                'estimator__min_impurity_decrease': 0.30
            },
            target=target,
            target_hours=target_hours)


class GradientBoostedPredictor(SKPredictor):
    @classmethod
    def get_name(cls):
        return 'gbt'

    def __init__(self, target='UGA-C-POA-1-IRR', target_hours=tuple(np.arange(1, 25))):
        super().__init__(
            estimator=XGBRegressor(),
            parameter_grid={
                'estimator__learning_rate': np.arange(0.03, 0.07, 0.02),   # learning rate
                'estimator__n_estimators': [50, 100, 200, 250],            # number of boosting stages
                'estimator__max_depth': [3, 5, 10, 20],
            },
            default_params={
                'estimator__learning_rate': 0.05,
                'estimator__n_estimators': 200,
                'estimator__max_depth': 5,
            },
            target=target,
            target_hours=target_hours)
