import importlib
import json
import logging
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pickle5 as pickle

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pvlib
from pvlib import solarposition

from apollo import storage
from apollo.datasets import nam


logger = logging.getLogger(__name__)


def times_to_reftimes(times):
    '''Compute the reference times for forecasts at the given times.

    This attempts to get the reftimes for all forecasts that include this time.
    On the edge case, this may select one extra forecast per time.
    '''
    reftimes = pd.DatetimeIndex(times, name='reftime', tz='UTC').unique()
    a = reftimes.floor('6h').unique()
    b = a - pd.Timedelta('6h')
    c = a - pd.Timedelta('12h')
    d = a - pd.Timedelta('18h')
    e = a - pd.Timedelta('24h')
    f = a - pd.Timedelta('30h')
    g = a - pd.Timedelta('36h')
    return a.union(b).union(c).union(d).union(e).union(f).union(g)


def time_of_day(times):
    '''Compute time-of-day features.

    Arguments:
        times (numpy.ndarray like):
            A series of timestamps.

    Returns:
        pandas.DataFrame:
            A data frame with 2 data variables:
                - ``time_of_day_sin`` for the sin component of time-of-day
                - ``time_of_day_cos`` for the cosin component of time-of-day

            The data frame is indexed by the input timestamps.
    '''
    nanos = np.asarray(times, dtype='datetime64[ns]').astype('float32')
    seconds = nanos / 1e9
    days = seconds / 86400

    return pd.DataFrame({
        'time_of_day_sin': np.sin(days * 2 * np.pi),
        'time_of_day_cos': np.cos(days * 2 * np.pi),
    }, index=pd.DatetimeIndex(times, tz='UTC'))


def time_of_year(times):
    '''Compute time-of-year features.

    Arguments:
        times (numpy.ndarray like):
            A series of timestamps.

    Returns:
        pandas.DataFrame:
            A data frame with 2 data variables:
                - ``time_of_year_sin`` for the sin component of time-of-year.
                - ``time_of_year_cos`` for the cosin component of time-of-year.

            The data frame is indexed by the input timestamps.
    '''
    nanos = np.asarray(times, dtype='datetime64[ns]').astype('float32')
    seconds = nanos / 1e9
    days = seconds / 86400

    # Convert the data into day and year units.
    # The time-of-year is measured in Julian years, 365.25 days of 86400 seconds
    # each. These are not exactly equal to a Gregorian year (i.e. a year on the
    # western calendar) once you get into the crazy leap rules.
    years = days / 365.25

    return pd.DataFrame({
        'time_of_year_sin': np.sin(years * 2 * np.pi),
        'time_of_year_cos': np.cos(years * 2 * np.pi),
    }, index=pd.DatetimeIndex(times, tz='UTC'))


def daylight(times, lat, lon):
    '''Determine if the sun is above the horizon for the given times.

    The resulting series has a one hour leeway for both sunrise and sunset.

    Arguments:
        times (numpy.ndarray like):
            A series of timestamps.
        lat (float):
            The latitude.
        lon (float):
            The longitude.

    Returns:
        pandas.Series:
            A boolean series indexed by the times.
    '''
    times = pd.DatetimeIndex(times, name='time', tz='UTC')

    # Compute the time of the next sunrise, sunset, and transit (solar noon).
    # This computation assumes an altitude of sea-level and determined positions
    # based on the center of the sun. It could be adjusted for greater accuracy,
    # but the loss of generality probably isn't worth it.
    # (`rst` abreviates `rise_set_transit`)
    rst = solarposition.sun_rise_set_transit_ephem(times, lat, lon)

    # Daylight is when the next sunset preceeds the next sunrise.
    daylight = (rst.sunset < rst.sunrise)

    # Give ourselves leeway to account for the hour in which sunrise/set occurs.
    one_hour = pd.Timedelta(1, 'h')
    sunrise_edge = (rst.sunrise - rst.index < one_hour)
    sunset_edge = (rst.sunset - rst.index < one_hour)
    daylight |= sunrise_edge
    daylight |= sunset_edge

    # Ensure the series has a name.
    daylight.name = 'daylight'
    return daylight


def import_from_str(dotted):
    '''Import an object from a dotted import path.

    Arguments:
        dotted (str):
            A dotted import path. This string must contain at least one dot.
            Everything to the left of the last dot is interpreted as the import
            path to a Python module. The piece to the right of the last dot is
            the name of an object within that module.

    Returns:
        object:
            The object named by the dotted import path.
    '''
    (module, obj) = dotted.rsplit('.', 1)
    module = importlib.import_module(module)
    return getattr(module, obj)


def make_estimator(e):
    '''Cast to an estimator.

    If the input is a string, it is interpreted as a dotted import path to
    a constructor for the estimator. That constructor is called without
    arguments to create the estimator.

    If the input is a list, it is interpreted as a pipeline of transformers and
    estimators. Each element must be a pair ``(name, params)`` where ``name``
    is a dotted import path to a constructor, and ``params`` is a dict
    providing hyper parameters. The final step must be an estimator, and the
    intermediate steps must be transformers.

    If the input is an object, it is checked to contain ``fit`` and ``predict``
    methods and is used directly as an estimator.

    Otherwise this function raises an :class:`ValueError`.

    Returns:
        sklearn.base.BaseEstimator:
            The estimator.

    Raises:
        ValueError:
            The input could not be cast to an estimator.
    '''
    # If ``e`` is a dotted import path, import it then call it.
    if isinstance(e, str):
        ctor = import_from_str(e)
        estimator = ctor()

    # If it has a length, interpret ``e`` as a list of pipeline steps.
    elif hasattr(e, '__len__'):
        steps = []
        for (name, params) in e:
            ctor = import_from_str(name)
            step = ctor(**params)
            steps.append(step)
        estimator = make_pipeline(*steps)

    # Otherwise interpret ``e`` directly as an estimator.
    else:
        estimator = e

    # Ensure that it at least has `fit` and `predict`.
    try:
        getattr(estimator, 'fit')
        getattr(estimator, 'predict')
    except AttributeError:
        raise ValueError('could not cast into an estimator')

    return estimator


def from_template(template, **kwargs):
    '''Construct a model from a template.

    A template is a dictionary giving keyword arguments for the constructor
    :class:`apollo.models2.Model`. Alternativly, the dictionary may contain
    the key ``_ctor`` giving a dotted import path to an alternate constructor.

    The ``template`` argument may take several forms:

    :class:`dict`
        A dictionary is interpreted as a template directly.
    file-like object
        A file-like object is parsed as JSON.
    :class:`pathlib.Path`
        A path to a JSON file containing the template.
    :class:`str`
        A string that does not contain a path separator ('/' or '\') is
        interpreted as a named template. Otherwise it is interpretead as a path
        to a JSON file containing the template. Named templates can be listed
        with :func:`apollo.models2.list_templates`.

    Arguments:
        template (dict or str or pathlib.Path or io.IOBase):
            A template, template name, path to a template, or JSON file.
        **kwargs:
            Additional keyword arguments to pass to the model constructor.

    Returns:
        apollo.models2.Model:
            An untrained model.
    '''
    # Convert str to Path.
    if isinstance(template, str):
        if os.sep in template:
            template = Path(template)
        else:
            template = storage.get('models2') / 'templates' / f'{template}.json'

    # Convert Path to file-like.
    if isinstance(template, Path):
        template = template.open('r')

    # Convert file-like to dict.
    if hasattr(template, 'read'):
        template = json.load(template)

    # The kwargs override the template.
    template.update(kwargs)

    # Load from dict.
    logger.debug(f'using template: {template}')
    ctor = template.pop('_ctor', 'apollo.models2.Model')
    ctor = import_from_str(ctor)
    model = ctor(**template)
    return model


def list_templates():
    '''List the named templates.

    Untrained models can be constructed from these template names using
    :func:`apollo.models2.from_template`.

    Returns:
        list of str:
            The named templates.
    '''
    base = storage.get('models2') / 'templates'
    base.mkdir(parents=True, exist_ok=True)
    template_paths = base.glob('*.json')
    template_stems = [p.stem for p in template_paths]
    return template_stems


def list_models():
    '''List the trained models.

    Trained models can be constructed from these names using
    :func:`apollo.models2.load`.

    Returns:
        list of str:
            The trained models.
    '''
    base = storage.get('models2') / 'models'
    base.mkdir(parents=True, exist_ok=True)
    model_paths = base.glob('*.pickle')
    model_stems = [p.stem for p in model_paths]
    return model_stems


def _discard(l, value):
    '''Like :meth:``set.discard``, but for lists.
    '''
    try:
        l.remove(value)
    except ValueError:
        pass


class Model:
    def __init__(
        self,
        name=None,
        estimator='sklearn.linear_model.LinearRegression',
        features=nam.PLANAR_FEATURES,
        add_time_of_day=True,
        add_time_of_year=True,
        daylight_only=False,
        standardize=False,
        center=nam.ATHENS_LATLON,
        shape=12000,
    ):
        '''Construct a new model.

        Arguments:
            name (str or None):
                A name for the estimator. If not given, a UUID is generated.
            estimator (sklearn.base.BaseEstimator or str):
                A Scikit-learn estimator to generate predictions. If a string
                is passed, it is interpreted as a dotted import path to a
                constructor for the estimator.
            features (list of str):
                The NAM features to use for prediction.
            add_time_of_day (bool):
                If true, compute time-of-day features.
            add_time_of_year (bool):
                If true, compute time-of-year features.
            daylight_only (bool):
                If true, timestamps which occur at night are ignored during
                training and are always predicted to be zero.
            standardize (bool):
                If true, standardize the data before sending it to the
                estimator. This transform is not applied to the computed
                time-of-day and time-of-year features.
            center (pair of float):
                The center of the geographic, as a latitude-longited pair.
            shape (float or pair of float):
                The height and width of the geographic area, measured in meters.
                If a scalar, both height and width are the same size.
        '''
        self.name = name or uuid.uuid4()
        self.estimator = make_estimator(estimator)
        self.features = list(features)
        self.add_time_of_day = bool(add_time_of_day)
        self.add_time_of_year = bool(add_time_of_year)
        self.daylight_only = bool(daylight_only)
        self.standardize = bool(standardize)
        self.center = center
        self.shape = shape

        # The names of the output columns, derived from the targets.
        self.columns = None

        # The standardizer, if one is to be used.
        self.std_scaler = StandardScaler(copy=False)

    def load_data(self, times, dedupe_strategy='best'):
        '''Load input data for the given times.

        Arguments:
            times (np.ndarray like):
                A series of timestamps.
            dedupe_strategy (str or int):
                The strategy for selecting between duplicate forecasts.
                **TODO:** Better documentation.

        Returns:
            pandas.DataFrame:
                A data fram indexed by the forecast time.
        '''
        times = pd.DatetimeIndex(times, name='time', tz='UTC')
        times = times.floor('1h').unique()

        # Load the xarray data.
        logger.debug('load: loading netcdf')
        reftimes = times_to_reftimes(times)
        data = nam.open(reftimes, on_miss='skip')
        data = data[self.features]
        data = data.astype('float32')

        # Select geographic area.
        logger.debug('load: slicing geographic area')
        data = nam.slice_geo(data, center=self.center, shape=self.shape)

        # Create a data frame.
        # This will have a multi-index of `(reftime, forecast, x, y, *z)`,
        # where `*z` is all of the different z-axes in the dataset.
        logger.debug('load: converting to dataframe')
        data = data.to_dataframe().drop(['lat', 'lon'], axis=1)

        # Replace `reftime` and `forecast` levels with `time`.
        logger.debug('load: reindex by time')
        old_index = data.index
        data = data.set_index('time', append=True)
        data = data.reorder_levels(['time', *old_index.names])
        data = data.reset_index('reftime', drop=True)
        data = data.reset_index('forecast', drop=False)

        # Filter to only the times requested.
        data = data.loc[times]

        # Handle duplicates.
        logger.debug(f'load: selecting forecast hour (dedupe_strategy={dedupe_strategy})')
        if dedupe_strategy == 'best':
            data = data.groupby(data.index) \
                .apply(lambda g: g[g.forecast == g.forecast.min()]) \
                .droplevel(0)
        elif isinstance(dedupe_strategy, int) and dedupe_strategy < 6:
            delta = pd.Timedelta('6h')
            lo = delta * dedupe_strategy
            hi = lo + delta
            data = data.groupby(data.index) \
                .apply(lambda g: g[(lo <= g.forecast) & (g.forecast < hi)]) \
                .droplevel(0)
        else:
            raise ValueError(f'invalid dedupe_strategy {repr(dedupe_strategy)}')

        # We no longer need the forecast hour.
        data = data.drop('forecast', axis=1)

        # Drop rows with NaNs or infinities.
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna()

        # Unstack until we're indexed by time alone, then flatten the columns.
        logger.debug('load: unstacking geographic indices')
        while 2 < len(data.index.levels):
            data = data.unstack()
        data = data.unstack()
        data.columns = data.columns.to_flat_index()

        # Ensure time is localized to UTC.
        data.index = data.index.tz_localize('UTC')

        # Add time-of-day and time-of-year features.
        if self.add_time_of_day:
            logger.debug('load: computing time-of-day')
            data = data.join(time_of_day(data.index))
        if self.add_time_of_year:
            logger.debug('load: computing time-of-year')
            data = data.join(time_of_year(data.index))

        return data

    def fit(self, targets):
        logger.debug('fit: downsampling targets to 1H frequency')
        targets = targets.resample('1H').mean()

        if self.daylight_only:
            logger.debug('fit: filter out night times')
            times = targets.index
            (lat, lon) = self.center
            targets = targets[daylight(times, lat, lon)]

        logger.debug('fit: loading forecasts')
        data = self.load_data(targets.index, dedupe_strategy='best')

        logger.debug('fit: intersecting forecasts with targets')
        index = data.index.intersection(targets.index)
        data = data.loc[index]
        targets = targets.loc[index]

        if self.standardize:
            logger.debug('fit: standardizing forecast data')
            cols = list(data.columns)
            _discard(cols, 'time_of_day_cos')
            _discard(cols, 'time_of_day_sin')
            _discard(cols, 'time_of_year_cos')
            _discard(cols, 'time_of_year_sin')
            raw_data = data[cols]
            self.std_scaler.fit(raw_data)

        logger.debug('fit: fitting estimator')
        self.columns = list(targets.columns)
        self.estimator.fit(data.to_numpy(), targets.to_numpy())
        return self

    def predict(self, times, dedupe_strategy='best'):
        times = pd.DatetimeIndex(times, name='time', tz='UTC')
        times = times.sort_values().unique()

        logger.debug('predict: loading forecasts')
        data = self.load_data(times, dedupe_strategy=dedupe_strategy)

        if self.standardize:
            logger.debug('fit: standardizing forecast data')
            cols = list(data.columns)
            _discard(cols, 'time_of_day_cos')
            _discard(cols, 'time_of_day_sin')
            _discard(cols, 'time_of_year_cos')
            _discard(cols, 'time_of_year_sin')
            raw_data = data[cols]
            data[cols] = self.std_scaler.transform(raw_data)

        logger.debug('predict: executing the model')
        predictions = self.estimator.predict(data.to_numpy())
        predictions = pd.DataFrame(predictions, data.index, self.columns)

        if self.daylight_only:
            logger.debug('predict: setting night time to zero')
            times = predictions.index
            (lat, lon) = self.center
            night = not daylight(times, lat, lon)
            predictions.loc[night, :] = 0

        return predictions

    def save(self, path=None):
        '''Persist a model to disk.

        Arguments:
            model (apollo.models2.Model):
                The model to persist.
            path (str or pathlib.Path or None):
                The path at which to save the model. The default is a path
                within your ``$APOLLO_DATA`` directory. Models saved to the
                default path can be loaded by name.

        Returns:
            pathlib.Path:
                The path at which the model was saved.
        '''
        if path is None:
            base = storage.get('models2') / 'models'
            base.mkdir(parents=True, exist_ok=True)
            path = base / f'{self.name}.pickle'
        else:
            path = Path(path)

        fd = path.open('wb')
        pickle.dump(self, fd, protocol=5)
        return path

    @classmethod
    def load(path_or_name):
        '''Load a model from disk.

        Arguments:
            path_or_name (str or pathlib.Path):
                If the argument is a string that does not contain a path
                separator, it is interpreted as the name of a model stored in
                the ``$APOLLO_DATA`` directory. Otherwise, it is interpreted as
                a path to the model.

        Returns:
            apollo.models2.Model:
                The model.
        '''
        if isinstance(path, str) and os.sep not in path:
            base = storage.get('models2') / 'models'
            base.mkdir(parents=True, exist_ok=True)
            path = base / f'{name}.pickle'
        else:
            path = Path(path)

        fd = path.open('rb')
        return pickle.load(fd)
