import json
import pickle
from pathlib import Path

import pandas as pd

from pvlib.forecast import GFS, HRRR, RAP, NAM, NDFD

from apollo import timestamps
from apollo.models.base import Model


class PVLibModel(Model):
    '''A model using :mod:`pvlib` for predictions.
    '''

    def __init__(self, name, forecast_model='NAM', forecast_hours=25,
            target='GHI', lat=33.9052058, lon=-83.382608):
        '''Construct a new :class:`PVLibModel`.

        Arguments:
            name (str):
                A name for the model instance.
            forecast_model (str or pvlib.forecast.ForecastModel):
                The pvlib forecast model to use.
            forecast_hours (int):
                The forecast hours. The model will output this many forecasts
                at one hour intervals, starting at the 0-hour forecast.
            target (str):
                The case-insensitive name of the feature to output. This must
                be one of the features in a pvlib forecast dataframe, including:
                - ``'DNI'`` for direct normal irradiance.
                - ``'DHI'`` for diffuse horizontal irradiance.
                - ``'GHI'`` for global horizontal irradiance.
            lat (float):
                The latitude of the forecast. The default is for Athens, GA.
            lon (float):
                The longitude of the forecast. The default is for Athens, GA.
        '''
        self._name = name
        self.forecast_hours = forecast_hours
        self._target = target
        self.lat = lat
        self.lon = lon

        if forecast_model == 'GFS':
            self._forecast_model_str = 'GFS'
            self.forecast_model = GFS()
        elif forecast_model == 'HRRR':
            self._forecast_model_str = 'HRRR'
            self.forecast_model = HRRR()
        elif forecast_model == 'RAP':
            self._forecast_model_str = 'RAP'
            self.forecast_model = RAP()
        elif forecast_model == 'NAM':
            self._forecast_model_str = 'NAM'
            self.forecast_model = NAM()
        elif forecast_model == 'NDFD':
            self._forecast_model_str = 'NDFD'
            self.forecast_model = NDFD()
        else:
            self._forecast_model_str = '_other'
            self.forecast_model = forecast_model

    @property
    def name(self):
        '''The name of this model, as set by the constructor.
        '''
        return self._name

    def save(self, path):
        '''Save this model to a path.

        Arguments:
            path (pathlib.Path):
                The directory in which to store model.
        '''
        path = Path(path)

        # These parameters need to be serialized.
        params = {
            'name': self.name,
            'forecast_model': self._forecast_model_str,
            'forecast_hours': self.forecast_hours,
            'target': selt._target,
            'lat': self.lat,
            'lon': self.lon,
        }

        # Save the parameters in a JSON file.
        param_path = path / 'params.json'
        with param_path.open('w') as fd:
            json.dump(params, fd)

        # If the forecast model is "_other", we also pickle the model.
        if params['forecast_model'] == '_other':
            model_path = path / 'model.pickle'
            with model_path.open('w') as fd:
                pickle.dump(params, fd)

    @classmethod
    def load(cls, path):
        '''Load an instance of :class:`PVLibModel` from a path.

        Arguments:
            path (pathlib.Path):
                The directory in which the model is stored.

        Returns:
            Model:
                An instance of the model.
        '''
        path = Path(path)

        # Load the parameters from a JSON file.
        param_path = path / 'params.json'
        with param_path.open('r') as fd:
            params = json.load(params, fd)

        # If the forecast model is "_other", we also unpickle the model.
        if params['forecast_model'] == '_other':
            model_path = path / 'model.pickle'
            with model_path.open('r') as fd:
                forecast_model = pickle.load(params, fd)
                params['forecast_model'] = forecast_model

        return cls(**params)

    def fit(self, first, last):
        '''Does nothing because this is not a learning model.
        '''
        pass

    def forecast(self, reftime):
        '''Make a forecast for the given reftime.

        This returns forecasts at one-hour intervals starting at the reftime.

        Arguments:
            reftime (pandas.Timestamp):
                The reference time of the forecast.

        Returns:
            forecast (pandas.Dataframe):
                A dataframe of forecasts with a :class:`~pandas.DatetimeIndex`.
                The index gives the timestamp of the forecast hours, and the
                only column corresponds to the target variable being forecast.
        '''
        start = timestamps.utc_timestamp(reftime).floor('6h')
        end = start + pd.Timedelta(1, 'h') * (self.forecast_hours - 1)

        data = self.forecast_model.get_processed_data(
            latitude=self.lat,
            longitude=self.lon,
            start=start,
            end=end,
        )

        forecast_times = pd.date_range(
            start = start,
            periods = self.forecast_hours,
            freq = 'h',
        )

        # The pvlib API makes no guarantees about the indexes of the data.
        # We can use a combination of ``combine_first``, ``interpolate``, and
        # ``reindex`` to ensure the data matches our index.
        target = self._target.lower()
        null_series = pd.DataFrame({target: None}, forecast_times)
        return (data[[target]]
            .combine_first(null_series)
            .interpolate()
            .reindex(forecast_times)
        )

    @property
    def target(self):
        '''The name of the variable that this model targets.

        Returns:
            str: name of the target variable.
        '''
        return self._target.upper()

    @property
    def target_hours(self):
        '''Forecast hours for which this model makes predictions

        Returns:
            tuple: hours targeted by this model.
        '''
        return tuple(i for i in range(self.forecast_hours))
