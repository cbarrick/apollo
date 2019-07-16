import logging
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import pickle5 as pickle

import apollo
from apollo.models import make_estimator


logger = logging.getLogger(__name__)


class Model(ABC):
    '''Base class for Apollo models.

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

    - :meth:`load_data(index, **kwargs)`: Models are responsible for loading
      their own feature data for training and prediction. This method recieves
      a :class:`pandas.Index` and returns some object representing the feature
      data. It may accept additional keyword arguments.

    - :meth:`fit(targets, **kwargs)`: Like Scikit-learn estimators, Apollo
      models must have a ``fit`` method for fitting the model. However, unlike
      Scikit-learn estimators, Apollo models have a single required argument,
      ``targets`` which must be a :class:`pandas.DataFrame`. Additional
      arguments should be forwarded to :meth:`load_data`.

    - :meth:`predict(index, **kwargs)`: Again like Scikit-learn estimators,
      Apollo models must have a ``predict`` method for generating predictions.
      The input to this method is a :class:`pandas.Index` for the resulting
      prediction. The return value is a :class:`pandas.DataFrame` using that
      index and columns like the target data frame passed to :meth:`fit`.
      Additional arguments should be forwarded to :meth:`load_data`.

    This base class provides default implementations of :meth:`fit` and
    :meth:`predict`, however using these requires you to understand a handfull
    of lower-level pieces.

    - :attr:`estimator`: The default implementation wraps a Scikit-learn style
      estimator to perform the actual predictions. It recieves as input the
      values produced by :meth:`preprocess` (described below). You **must**
      provide an estimator attribute if you use the default :meth:`fit` or
      :meth:`predict`.

    - :meth:`preprocess(features, targets=None, fit=False)`: This method
      transforms the "structured data" returned by :meth:`load_data` into the
      "raw data" passed to the estimator. The ``targets`` argument may not be
      given, and the ``fit`` argument is true when preprocessing for :meth:`fit`
      and false when preprocessing for :meth:`predict`. The return value is a
      pair ``(raw_features, raw_targets)`` where ``raw_targets`` is None when
      ``targets`` is None. A default implementation is provided which simply
      passes the feature and target data through :func:`numpy.asanyarray`.

    - :meth:`postprocess(raw_predictions, index)`: This method transforms the
      "raw predictions" returned by the estimator into a fully-fledged
      :class:`pandas.DataFrame`. The default implementation simply delegates
      to the ``DataFrame`` constructor.
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
            raw_features:
                Processed feature data.
            raw_targets:
                Processed target data or ``None`` if no target data was given.
        '''
        raw_features = np.asanyarray(features)
        raw_targets = np.asanyarray(raw_targets) or None
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
        raw_data, raw_targets = self.preprocess(data, targets, fit=True)
        self.estimator.fit(raw_data, raw_targets)
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
        raw_data, _ = self.preprocess(data)
        raw_predictions = self.estimator.predict(raw_data)
        prediction = self.postprocess(raw_predictions, index)
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

        fd = path.open('wb')
        pickle.dump(self, fd, protocol=5)
        return path


class IrradianceModel(Model):
    '''A base class for irradiance modeling.

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
    ):
        '''Construct a new model.

        Keyword Arguments:
            name (str or None):
                A name for the estimator. If not given, a UUID is generated.
            estimator (sklearn.base.BaseEstimator or str or list):
                A Scikit-learn estimator to generate predictions. It is
                interpreted by :func:`apollo.models.make_estimator`.
            standardize (bool):
                If true, standardize the data before sending it to the
                estimator. This transform is not applied to the computed
                time-of-day and time-of-year features.
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
        super().__init__(name, estimator)

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
            self.columns = list(targets.columns)
        elif targets is not None:
            assert set(targets.columns) == set(self.columns)

        # Drop NaNs and infinities.
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if targets is not None:
            targets = targets.replace([np.inf, -np.inf], np.nan).dropna()

        # We only support 1-hour frequencies.
        # For overlapping targets, take the mean.
        if targets is not None:
            targets = targets.groupby(targets.index.floor(1, 'h')).mean()

        # Ignore targets at night (optionally).
        if targets is not None and self.daylight_only:
            times = targets.index
            (lat, lon) = self.center
            targets = targets[apollo.is_daylight(times, lat, lon)]

        # The indices for the data and targets may not match.
        # We can only consider their intersection.
        if targets is not None:
            index = data.index.intersection(targets.index)
            data = data.loc[index]
            targets = targets.loc[index]
        else:
            index = data.index

        # Scale the feature data (optionally).
        if self.standardize:
            cols = list(data.columns)
            raw_data = data[cols].to_numpy()
            if fit: self.feature_scaler.fit(raw_data)
            data[cols] = self.feature_scaler.transform(raw_data)

        # Scale the target data.
        # Unlike the features, we _always_ scale the targets.
        if targets is not None:
            cols = self.columns
            raw_targets = targets[cols].to_numpy()
            if fit: self.target_scaler.fit(raw_targets)
            targets[cols] = self.target_scaler.transform(raw_targets)

        # Compute additional features (optionally).
        if self.add_time_of_day: data = data.join(apollo.time_of_day(index))
        if self.add_time_of_year: data = data.join(apollo.time_of_year(index))

        # We always return both, even if targets was not given.
        # We must return numpy arrays.
        return data.to_numpy(), targets.to_numpy()

    def postprocess(self, times, raw_predictions):
        '''
        '''
        # Reconstruct the data frame.
        cols = self.cols
        index = apollo.DatetimeIndex(times, name='time')
        predictions = pd.DataFrame(raw_predictions, index=index, columns=cols)

        # Unscale the predictions.
        predictions[cols] = self.target_scaler.inverse_transform(raw_predictions)

        # Set overnight predictions to zero (optionally).
        if self.daylight_only:
            (lat, lon) = self.center
            night = not apollo.is_daylight(index, lat, lon)
            predictions.loc[night, :] = 0

        return predictions
