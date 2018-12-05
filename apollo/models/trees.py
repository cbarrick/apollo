import abc
import numpy as np
import pandas as pd
import pathlib
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from apollo.datasets.solar import SolarDataset
from apollo.models.base import Model


class TreeModel(Model, abc.ABC):
    def __init__(self, name=None, data_kwargs=None, **kwargs):
        ts = pd.Timestamp('now')
        self._name = name or f'dtree@{ts.isoformat()}'
        data_kwargs = data_kwargs or {}
        default_data_kwargs = {
            'lag': 0,
            'target': 'UGA-C-POA-1-IRR',
            'target_hours': tuple(np.arange(1, 25)),
            'standardize': True
        }
        # self.data_kwargs will be a merged dictionary with values from `data_kwargs` replacing default values
        self.data_kwargs = {**default_data_kwargs, **data_kwargs}

        # self.model_kwargs will be a merged dictionary with values from `kwargs` replacing default values
        self.model_kwargs = {**self.default_hyperparams, **kwargs}
        self.model = None

    @property
    @abc.abstractmethod
    def estimator(self):
        ''' Estimator that conforms to the scikit-learn API
        '''
        pass

    @property
    @abc.abstractmethod
    def default_hyperparams(self):
        ''' Default hyperparameters to use with this model's estimator
        '''
        pass

    @property
    def name(self):
        return self._name

    @classmethod
    def load(cls, path):
        name = path.name
        with open(path / 'data_args.pickle', 'rb') as data_args_file:
            data_kwargs = pickle.load(data_args_file)
        with open(path / 'model_args.pickle', 'rb') as model_args_file:
            model_kwargs = pickle.load(model_args_file)
        model = cls(name=name, data_kwargs=data_kwargs, **model_kwargs)
        model.model = joblib.load(path / 'regressor.joblib')

        return model

    def save(self, path):
        if not self.model:
            raise ValueError('Model has not been trained.  Ensure `model.fit` is called before `model.save`.')

        # serialize the trained model
        joblib.dump(self.model, path / 'regressor.joblib')

        # serialize kwargs
        with open(path / 'data_args.pickle', 'wb') as outfile:
            pickle.dump(self.data_kwargs, outfile)
        with open(path / 'model_args.pickle', 'wb') as outfile:
            pickle.dump(self.model_kwargs, outfile)

    def fit(self, first, last):
        ds = SolarDataset(first, last, **self.data_kwargs)
        x, y = ds.tabular()
        x = np.asarray(x)
        y = np.asarray(y)
        self.estimator.set_params(**self.model_kwargs)
        model = MultiOutputRegressor(estimator=self.estimator, n_jobs=1)
        model.fit(x, y)
        self.model = model
        # save standardization parameters
        self.data_kwargs['standardize'] = (ds.mean, ds.std)

    def forecast(self, reftime):
        data_kwargs = dict(self.data_kwargs)
        data_kwargs['target'] = None

        reftime = pd.Timestamp(reftime)

        ds = SolarDataset(reftime, reftime + pd.Timedelta(6, 'h'), **data_kwargs)
        x = ds.tabular()
        x = np.asarray(x)

        y = self.model.predict(x)[0]
        index = [reftime + pd.Timedelta(1, 'h') * n for n in data_kwargs['target_hours']]
        series = pd.Series(y, index, name='predictions')
        return series


class DecisionTree(TreeModel):
    @property
    def estimator(self):
        return DecisionTreeRegressor()

    @property
    def default_hyperparams(self):
        return {
            'splitter': 'best',
            'max_depth': 20,
            'min_impurity_decrease': 0.25
        }


class RandomForest(TreeModel):
    @property
    def estimator(self):
        return RandomForestRegressor()

    @property
    def default_hyperparams(self):
        return {
            'n_estimators': 100,
            'max_depth': 50,
            'min_impurity_decrease': 0.30
        }


class GradientBoostedTrees(TreeModel):
    @property
    def estimator(self):
        return XGBRegressor()

    @property
    def default_hyperparams(self):
        return {
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': 5,
        }