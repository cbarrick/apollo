import json
import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import pickle5 as pickle

import sklearn
from sklearn.preprocessing import StandardScaler

import apollo
from apollo import metrics


logger = logging.getLogger(__name__)


def list_templates():
    '''List the model templates in the Apollo database.

    Untrained models can be constructed from these template names using
    :func:`apollo.models.make_model`.

    Returns:
        list of str:
            The named templates.
    '''
    base = apollo.path('templates')
    base.mkdir(parents=True, exist_ok=True)
    template_paths = base.glob('*.json')
    template_stems = [p.stem for p in template_paths]
    return template_stems


def list_models():
    '''List trained models in the Apollo database.

    Trained models can be constructed from these names using
    :func:`apollo.models.load_named_model`.

    Returns:
        list of str:
            The trained models.
    '''
    base = apollo.path('models')
    base.mkdir(parents=True, exist_ok=True)
    model_paths = base.glob('*.model')
    model_stems = [p.stem for p in model_paths]
    return model_stems


def make_estimator(e):
    '''An enhanced version of :func:`sklearn.pipeline.make_pipeline`.

    If the input is a string, it is interpreted as a dotted import path to
    a constructor for the estimator. That constructor is called without
    arguments to create the estimator.

    If the input is a list, it is interpreted as a pipeline of transformers and
    estimators. Each element must be a pair ``(name, params)`` where ``name``
    is a dotted import path to a constructor, and ``params`` is a dict
    providing hyper parameters. The final step must be an estimator, and the
    intermediate steps must be transformers.

    If the input is any other object, it is checked to contain ``fit`` and
    ``predict`` methods and is assumed to be the estimator.

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
        ctor = apollo._import_from_str(e)
        estimator = ctor()

    # If it has a length, interpret ``e`` as a list of pipeline steps.
    elif hasattr(e, '__len__'):
        steps = []
        for (name, params) in e:
            if isinstance(ctor, str):
                ctor = apollo._import_from_str(name)
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


def load_model_from(stream):
    '''Load a model from a file-like object.

    Arguments:
        stream (io.IOBase):
            A readable, binary stream containing a serialized model.

    Returns:
        apollo.models.Model:
            The model.
    '''
    model = pickle.load(stream)
    assert isinstance(model, Model), f'{path} is not an Apollo model'
    return model


def load_model_at(path):
    '''Load a model that is serialized at some path.

    Arguments:
        path (str or pathlib.Path):
            A path to a model.

    Returns:
        apollo.models.Model:
            The model.
    '''
    path = Path(path)
    stream = path.open('rb')
    return load_model_from(stream)


def load_model(name):
    '''Load a model from the Apollo database.

    Models in the Apollo database can be listed with :func:`list_models`
    or from the command line with ``apollo ls models``.

    Arguments:
        name (str):
            The name of the model.

    Returns:
        apollo.models.Model:
            The model.

    Raises:
        FileNotFoundError:
            No model exists in the database with the given name.
    '''
    path = apollo.path(f'models/{name}.model')
    if not path.exists():
        raise FileNotFoundError(f'No such model: {name}')
    return load_model_at(path)


def make_model_from(template, **kwargs):
    '''Construct a model from a template.

    A template is a dictionary giving keyword arguments for the constructor
    :class:`apollo.models.Model`. Alternativly, the dictionary may contain
    the key ``_cls`` giving a dotted import path to an alternate constructor.

    The ``template`` argument may take several forms:

    :class:`dict`
        A dictionary is interpreted as a template directly.
    :class:`io.IOBase`
        A file-like object containing JSON is parsed into a template.
    :class:`pathlib.Path` or :class:`str`
        A path to a JSON file containing the template.

    Arguments:
        template (dict or str or pathlib.Path or io.IOBase):
            A template dictionary or path to a template file.
        **kwargs:
            Additional keyword arguments to pass to the model constructor.

    Returns:
        apollo.models.Model:
            An untrained model.
    '''
    # Convert str to Path.
    if isinstance(template, str):
        template = Path(template)

    # Convert Path to file-like.
    if isinstance(template, Path):
        template = template.open('r')

    # Convert file-like to dict.
    if hasattr(template, 'read'):
        template = json.load(template)

    # The kwargs override the template.
    template.update(kwargs)

    # Determine which class to instantiate.
    cls = template.pop('_cls', 'apollo.models.NamModel')
    cls = apollo._import_from_str(cls)

    # Load from dict.
    logger.debug(f'using template: {template}')
    model = cls(**template)
    return model


def make_model(template_name, **kwargs):
    '''Construct a model from named template in the Apollo database.

    Templates in the Apollo database can be listed with :func:`list_templates`
    or from the command line with ``apollo ls templates``.

    Arguments:
        template_name (str):
            The name of a template in the Apollo database.
        **kwargs:
            Additional keyword arguments to pass to the model constructor.

    Returns:
        apollo.models.Model:
            An untrained model.

    Raises:
        FileNotFoundError:
            No template exists in the database with the given name.
    '''
    path = apollo.path(f'templates/{template_name}.json')
    if not path.exists():
        raise FileNotFoundError(f'No such template: {template_name}')
    return make_model_from(path)


class Model(ABC):
    '''Base class for all Apollo models.

    In general, an Apollo model is an object which makes predictions, similar
    to a Scikit-learn estimator. The features being predicted are determined by
    the target data when the model is fit, and a model must be fit before it can
    make predictions.

    The :class:`NamModel` provides a fully functioning model that makes
    predictions from NAM forecasts and provides several useful preprocesses for
    predicting solar irradiance.

    Unlike a Scikit-learn estimator, an Apollo model only recieves target data
    when it is fit; the model is responsible for loading its own feature data
    for training and prediction. Also unlike a Scikit-learn estimator, the
    methods :meth:`fit` and :meth:`predict` deal in Pandas data frames and
    indices rather than raw Numpy arrays.

    All models must have the following attributes and methods:

    - :attr:`name`: Each model must provide a name. This is used to when saving
      and loading models to and from the Apollo database. The name is accessed
      as an attribute.

    - :meth:`load_data`: Models are responsible for loading their own feature
      data for training and prediction. This method recieves a Pandas index and
      returns some object representing the feature data. It may accept
      additional, optional arguments.

    - :meth:`fit`: Like Scikit-learn estimators, Apollo models must have a
      ``fit`` method for training the model. However, unlike Scikit-learn
      estimators, Apollo models have a single required argument, a target
      DataFrame to fit against. Additional arguments should be forwarded to
      :meth:`load_data`.

    - :meth:`predict`: Again like Scikit-learn estimators, Apollo models must
      have a ``predict`` method for generating predictions. The input to this
      method is a Pandas index for the resulting prediction. The return value
      is a DataFrame using that index and columns like the target data frame
      passed to :meth:`fit`. Additional arguments should be forwarded to
      :meth:`load_data`.

    - :meth:`score`: All models have a score method, but unlike Scikit-learn,
      this method produces a data frame rather than a scalar value. The data
      frame has one column for each column in th training data, and each row
      gives a different metric. It is not specified what metrics should be
      computed nor how they should be interpreted. The default implementation
      delegates to :func:`apollo.metrics.all`.

    This base class provides default implementations of :meth:`fit` and
    :meth:`predict`, however using these requires you to understand a handfull
    of lower-level pieces.

    - :attr:`estimator`: The default implementation wraps a Scikit-learn style
      estimator to perform the actual predictions. It recieves as input the
      values produced by :meth:`preprocess` (described below). You **must**
      provide an estimator attribute if you use the default :meth:`fit` or
      :meth:`predict`.

    - :meth:`preprocess`: This method transforms the "structured data" returned
      by :meth:`load_data` and "structured targets" provided by the user into
      the "raw data" and "raw targets" passed to the estimator. A default
      implementation is provided that passes the input to the
      :class:`pandas.DataFrame` constructor. The values returned by the
      preprocess method are cast to numpy arrays with :func:`numpy.asanyarray`
      before being sent to the estimator.

    - :meth:`postprocess`: This method transforms the "raw predictions"
      returned by the estimator into a fully-fledged DataFrame. The default
      implementation simply delegates to the DataFrame constructor.
    '''

    def __init__(
        self, *,
        name=None,
        estimator='sklearn.linear_model.LinearRegression',
    ):
        '''Initialize a model.

        Subclasses which do not use the default :meth:`fit` and :meth:`predict`
        need not call this constructor.

        keyword Arguments:
            name (str or None):
                A name for the estimator. If not given, a UUID is generated.
            estimator (sklearn.base.BaseEstimator or str or list):
                A Scikit-learn estimator to generate predictions. It is
                interpreted by :func:`apollo.models.make_estimator`.
        '''
        self._name = str(name or uuid.uuid4())
        self._estimator = make_estimator(estimator)

    @property
    def name(self):
        '''The name of the model.
        '''
        return self._name

    @property
    def estimator(self):
        '''The underlying estimator.
        '''
        return self._estimator

    @abstractmethod
    def load_data(self, index, **kwargs):
        '''Load structured feature data according to some index.

        Arguments:
            index (pandas.Index):
                The index of the data.
            **kwargs:
                Implementations may accept additional, optional arguments.

        Returns:
            features:
                Structured feature data, typically a :class:`pandas.DataFrame`
                or a :class:`xarray.Dataset`.
        '''
        pass

    def preprocess(self, features, targets=None, fit=False):
        '''Convert structured data into raw data for the estimator.

        The default implementation passes both the features and targets to
        :func:`numpy.asanyarray`.

        Arguments:
            features:
                Structured data returned by :meth:`load_data`.
            targets:
                The target data passed into :meth:`fit`.
            fit (bool):
                If true, fit lernable transforms against this target data.

        Returns:
            pair of pandas.DataFrame:
                A pair of data frames ``(raw_features, raw_targets)`` containing
                processed feature data and processed target data respectivly.
                The ``raw_targets`` will be ``None`` if ``targets`` was None.
        '''
        raw_features = pd.DataFrame(features)
        raw_targets = pd.DataFrame(raw_targets) if targets is not None else None
        return raw_features, raw_targets

    def postprocess(self, raw_predictions, index):
        '''Convert raw predictions into a :class:`pandas.DataFrame`.

        The default implementation simply delegates to the
        :class:`pandas.DataFrame` constructor.

        Arguments:
            raw_predictions:
                The output of ``self.estimator.predict``.
            index (pandas.Index):
                The index of the resulting data frame.

        Returns:
            pandas.DataFrame:
                The predictions.
        '''
        return pd.DataFrame(raw_predictions, index=index)

    def fit(self, targets, **kwargs):
        '''Fit the models to some target data.

        Arguments:
            targets (pandas.DataFrame):
                The data to fit against.
            **kwargs:
                Additional arguments are forwarded to :meth:`load_data`.

        Returns:
            Model:
                self
        '''
        data = self.load_data(targets.index, **kwargs)
        data, targets = self.preprocess(data, targets, fit=True)
        logger.debug('fit: casting to numpy')
        data = np.asanyarray(data)
        targets = np.asanyarray(targets)
        logger.debug('fit: fitting estimator')
        self.estimator.fit(data, targets)
        return self

    def predict(self, index, **kwargs):
        '''Generate a prediction from this model.

        Arguments:
            index (pandas.Index):
                Make predictions for this index.
            **kwargs:
                Additional arguments are forwarded to :meth:`load_data`.

        Returns:
            pandas.DataFrame:
                A data frame of predicted values.
        '''
        data = self.load_data(index, **kwargs)
        data, _ = self.preprocess(data)
        index = data.index  # The preprocess step may change the index.
        logger.debug('predict: casting to numpy')
        data = np.asanyarray(data)
        logger.debug('predict: executing estimator')
        predictions = self.estimator.predict(data)
        predictions = self.postprocess(predictions, index)
        return predictions

    def save(self, path=None):
        '''Persist a model to disk.

        Arguments:
            path (str or pathlib.Path or None):
                The path at which to save the model. The default is a path
                within the Apollo database derived from the model's name.

        Returns:
            pathlib.Path:
                The path at which the model was saved.
        '''
        if path is None:
            path = apollo.path(f'models/{self.name}.model')
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path = Path(path)

        logger.debug(f'save: writing model to {path}')
        fd = path.open('wb')
        pickle.dump(self, fd, protocol=5)
        return path

    def score(self, targets):
        '''Score this model against some target values.

        Arguments:
            targets (pandas.DataFrame):
                The targets to compare against.

        Returns:
            pandas.DataFrame:
                A table of metrics.
        '''
        targets = targets.replace([np.inf, -np.inf], np.nan).dropna()
        predictions = self.predict(targets.index)

        n_missing = len(targets.index) - len(predictions.index)
        if n_missing != 0:
            logger.warning(f'missing {n_missing} predictions')
            targets = targets.reindex(predictions.index)

        scores = apollo.metrics.all(targets, predictions)
        scores.index.name = 'metric'
        return scores


class IrradianceModel(Model):
    '''Base class for irradiance modeling.

    This class implements :meth:`preprocess` and :meth:`postprocess` methods
    specifically for irradiance modeling. They require feature and target data
    to be both be a :class:`~pandas.DataFrame` indexed by timezone-aware
    :class:`~pandas.DatetimeIndex`.
    '''

    def __init__(
        self, *,
        standardize=False,
        add_time_of_day=True,
        add_time_of_year=True,
        daylight_only=False,
        center=None,
        **kwargs,
    ):
        '''Construct a new model.

        Keyword Arguments:
            name (str or None):
                A name for the estimator. If not given, a UUID is generated.
            estimator (sklearn.base.BaseEstimator or str or list):
                A Scikit-learn estimator to generate predictions. It is
                interpreted by :func:`apollo.models.make_estimator`.
            standardize (bool):
                If true, standardize the feature and target data before sending
                it to the estimator. This transform is not applied to the
                computed time-of-day and time-of-year features.
            add_time_of_day (bool):
                If true, compute time-of-day features.
            add_time_of_year (bool):
                If true, compute time-of-year features.
            daylight_only (bool):
                If true, timestamps which occur at night are ignored during
                training and are always predicted to be zero.
            center (pair of float):
                The center of the geographic area, as a latitude-longited pair.
                Used to compute the sunrise and sunset times. Required only
                when ``daylight_only`` is True.
        '''
        super().__init__(**kwargs)

        self.add_time_of_day = bool(add_time_of_day)
        self.add_time_of_year = bool(add_time_of_year)
        self.daylight_only = bool(daylight_only)
        self.standardize = bool(standardize)

        # The names of the output columns, derived from the targets.
        self.columns = None

        # The standardizers. The feature scaler may not be used.
        self.feature_scaler = StandardScaler(copy=False)
        self.target_scaler = StandardScaler(copy=False)

    def preprocess(self, data, targets=None, fit=False):
        '''Process feature data into a numpy array.
        '''
        # If we're fitting, we record the column names.
        # Otherwise we ensure the targets have the expected columns.
        if fit:
            logger.debug('preprocess: recording columns')
            self.columns = list(targets.columns)
        elif targets is not None:
            logger.debug('preprocess: checking columns')
            assert set(targets.columns) == set(self.columns)

        # Drop NaNs and infinities.
        logger.debug('preprocess: dropping NaNs and infinities')
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if targets is not None:
            targets = targets.replace([np.inf, -np.inf], np.nan).dropna()

        # We only support 1-hour frequencies.
        # For overlapping targets, take the mean.
        if targets is not None:
            logger.debug('preprocess: aggregating targets')
            targets = targets.groupby(targets.index.floor('1h')).mean()

        # Ignore targets at night (optionally).
        if targets is not None and self.daylight_only:
            logger.debug('preprocess: dropping night time targets')
            times = targets.index
            (lat, lon) = self.center
            targets = targets[apollo.is_daylight(times, lat, lon)]

        # The indices for the data and targets may not match.
        # We can only consider their intersection.
        if targets is not None:
            logger.debug('preprocess: joining features and targets')
            index = data.index.intersection(targets.index)
            data = data.loc[index]
            targets = targets.loc[index]
        else:
            index = data.index

        # Scale the feature data (optionally).
        if self.standardize:
            logger.debug('preprocess: scaling features')
            cols = list(data.columns)
            raw_data = data[cols].to_numpy()
            if fit: self.feature_scaler.fit(raw_data)
            data[cols] = self.feature_scaler.transform(raw_data)

        # Scale the target data (optionally).
        if self.standardize and targets is not None:
            logger.debug('preprocess: scaling targets')
            cols = self.columns
            raw_targets = targets[cols].to_numpy()
            if fit: self.target_scaler.fit(raw_targets)
            targets[cols] = self.target_scaler.transform(raw_targets)

        # Compute additional features (optionally).
        if self.add_time_of_day:
            logger.debug('preprocess: computing time-of-day')
            data = data.join(apollo.time_of_day(index))
        if self.add_time_of_year:
            logger.debug('preprocess: computing time-of-year')
            data = data.join(apollo.time_of_year(index))

        # We always return both, even if targets was not given.
        # We must return numpy arrays.
        logger.debug('preprocess: casting to numpy')
        return data, targets

    def postprocess(self, raw_predictions, index):
        '''
        '''
        # Reconstruct the data frame.
        logger.debug('postprocess: constructing data frame')
        index = apollo.DatetimeIndex(index, name='time')
        predictions = super().postprocess(raw_predictions, index)

        # Set the columns.
        cols = list(self.columns)
        assert len(predictions.columns) == len(cols)
        predictions.columns = pd.Index(cols)

        # Unscale the predictions.
        if self.standardize:
            logger.debug('postprocess: unscaling predictions')
            predictions[cols] = self.target_scaler.inverse_transform(raw_predictions)

        # Set overnight predictions to zero (optionally).
        if self.daylight_only:
            logger.debug('postprocess: setting night time to zero')
            (lat, lon) = self.center
            night = not apollo.is_daylight(index, lat, lon)
            predictions.loc[night, :] = 0

        return predictions

    def score(self, targets):
        '''Score this model against some target values.

        Arguments:
            targets (pandas.DataFrame):
                The targets to compare against.

        Returns:
            pandas.DataFrame:
                A table of metrics.
        '''
        targets = targets.replace([np.inf, -np.inf], np.nan).dropna()
        predictions = self.predict(targets.index)

        n_missing = len(targets.index) - len(predictions.index)
        if n_missing != 0:
            logger.warning(f'missing {n_missing} predictions')
            targets = targets.reindex(predictions.index)

        daynight_scores = apollo.metrics.all(targets, predictions)
        daynight_scores.index = daynight_scores.index + '_day_night'

        lat, lon = self.center
        is_daylight = apollo.is_daylight(targets.index, lat, lon)
        predictions = predictions[is_daylight]
        targets = targets[is_daylight]
        dayonly_scores = metrics.all(targets, predictions)
        dayonly_scores.index = dayonly_scores.index + '_day_only'

        scores = daynight_scores.append(dayonly_scores)
        scores.index.name = 'metric'
        return scores
