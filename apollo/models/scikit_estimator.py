import abc
import numpy as np
import pandas as pd
import pathlib
import pickle

from sklearn.externals import joblib
from sklearn.multioutput import MultiOutputRegressor

from apollo.datasets.solar import SolarDataset
from apollo.models.base import Model


class ScikitModel(Model, abc.ABC):
    ''' Abstract base class for models that use estimators conforming to the scikit-learn API
    '''
    def __init__(self, data_kwargs=None, model_kwargs=None, **kwargs):
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
        model_kwargs = model_kwargs or {}
        self.model_kwargs = {**self.default_hyperparams, **model_kwargs}
        self.model = None

        self._name = f'{self.__class__.__name__}@{ts.isoformat()}'
        if 'name' in kwargs:
            self._name = kwargs['name']

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
        df = pd.DataFrame(y, index=pd.DatetimeIndex(index), columns=[self.data_kwargs['target']])
        return df
