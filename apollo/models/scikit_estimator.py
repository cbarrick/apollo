import abc
import numpy as np
import pandas as pd
import pathlib
import pickle

from sklearn.externals import joblib
from sklearn.multioutput import MultiOutputRegressor

from apollo.datasets.solar import SolarDataset, PLANAR_FEATURES, ATHENS_LATLON
from apollo.models.base import Model


class ScikitModel(Model, abc.ABC):
    ''' Abstract base class for models using estimators from the sklearn API
    '''
    def __init__(self, name=None, **kwargs):
        ''' Initialize a ScikitModel

        Args:
            data_kwargs (dict or None):
                kwargs to be passed to the SolarDataset constructor
            model_kwargs (dict or None:
                kwargs to be passed to the scikit-learn estimator constructor
            **kwargs:
                other kwargs used for model initialization, such as model name
        '''
        ts = pd.Timestamp('now')
        # grab kwargs used to load data
        self.data_kwargs = {
            'feature_subset': PLANAR_FEATURES,
            'temporal_features': True,
            'center': ATHENS_LATLON,
            'geo_shape': (3, 3),
            'lag': 0,
            'forecast': 36,
            'target': 'UGABPOA1IRR',
            'target_hours': tuple(np.arange(1, 25)),
            'standardize': True
        }
        for key in self.data_kwargs:
            if key in kwargs:
                self.data_kwargs[key] = kwargs[key]

        # grab kwargs corresponding to model hyperparams
        self.model_kwargs = self.default_hyperparams
        for key in self.model_kwargs:
            if key in kwargs:
                self.model_kwargs[key] = kwargs[key]

        self.model = None

        self._name = name or f'{self.__class__.__name__}@{ts.isoformat()}'

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
        with open(path / 'kwargs.pickle', 'rb') as args_file:
            kwargs = pickle.load(args_file)
        model = cls(**kwargs)
        model.model = joblib.load(path / 'regressor.joblib')

        return model

    def save(self, path):
        if not self.model:
            raise ValueError('Model has not been trained. Ensure `model.fit` '
                             'is called before `model.save`.')

        # serialize the trained scikit-learn model
        joblib.dump(self.model, path / 'regressor.joblib')

        # serialize kwargs
        args = {
            'name': self.name
        }
        args = dict(args, **self.data_kwargs)
        args = dict(args, **self.model_kwargs)
        with open(path / 'kwargs.pickle', 'wb') as outfile:
            pickle.dump(args, outfile)

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
        index = [reftime + pd.Timedelta(1, 'h') * n
                 for n in data_kwargs['target_hours']]
        df = pd.DataFrame(y, index=pd.DatetimeIndex(index),
                          columns=[self.data_kwargs['target']])
        return df
