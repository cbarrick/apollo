import pandas as pd
import pathlib
import pickle

from apollo import casts
from apollo.datasets import ga_power
from apollo.datasets.solar import DEFAULT_TARGET, DEFAULT_TARGET_HOURS
from apollo.models.base import Model


class PersistenceModel(Model):
    ''' Predicts solar irradiance using the irradiance from 24 hours prior

    Persistence models often serve as baselines, since they are very simple to
    compute and require little data to use.
    There are a variety of models that may be referred to as
    "persistence models".  In this case, the model predicts the target variable
    at time `t` using the value of the target at time `t - 24`.

    Like many of the other Apollo models, the persistence model will fail if the
    required data is missing.  Unlike the other models, the persistence model
    does not require a recent NAM forecast, but rather an observation from the
    previous day.

    This model does not accept any hyperparameters.  All kwargs are forwarded
    to the data loader.
    '''
    def __init__(self, name=None, **kwargs):
        ''' Initialize a PersistenceModel

        Args:
            name (str):
                A descriptive name for the model.
            **kwargs:
                The keyword arguments forwarded to the data loader
        '''
        ts = casts.utc_timestamp('now')

        self.kwargs = kwargs
        self.data_args = kwargs
        self._name = f'PersistenceModel@{ts.isoformat()}' if name is None \
            else name

    @property
    def name(self):
        return self._name

    @property
    def target(self):
        return self.data_args['target'] \
            if 'target' in self.data_args \
            else DEFAULT_TARGET

    @property
    def target_hours(self):
        if 'target_hours' in self.data_args:
            try:
                return tuple(self.data_args['target_hours'])
            except TypeError:
                return self.data_args['target_hours'],
        else:
            try:
                return tuple(DEFAULT_TARGET_HOURS)
            except TypeError:
                return DEFAULT_TARGET_HOURS,

    @classmethod
    def load(cls, path):
        name = path.name
        with open(pathlib.Path(path) / 'kwargs.pickle', 'rb') as kwargs_file:
            kwargs = pickle.load(kwargs_file)
        model = cls(name=name, **kwargs)

        return model

    def save(self, path):
        # serialize kwargs
        with open(pathlib.Path(path) / 'kwargs.pickle', 'wb') as outfile:
            pickle.dump(self.kwargs, outfile)

    def fit(self, first, last):
        pass

    def forecast(self, reftime):
        reftime = casts.utc_timestamp(reftime)
        forecast_reach = max(*self.target_hours)  # maximum forecast hour
        past_values_start = reftime - pd.Timedelta(forecast_reach, 'h')
        past_values_end = reftime
        past_values = ga_power.open(
            self.target, start=past_values_start, stop=past_values_end)\
            .to_dataframe()

        index = [reftime + pd.Timedelta(1, 'h') * n for n in self.target_hours]
        predictions = []
        for timestamp in index:
            past_timestamp = timestamp - pd.Timedelta(24, 'h')
            if past_timestamp in past_values.index.values:
                past_val = past_values.loc[past_timestamp]
            else:
                past_val = 0
            predictions.append(past_val)

        df = pd.DataFrame(
            predictions, index=pd.DatetimeIndex(index), columns=[self.target])
        return df
