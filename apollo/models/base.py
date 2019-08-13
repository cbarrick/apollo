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
from apollo.models.functions import make_estimator, write_model, write_model_at


logger = logging.getLogger(__name__)


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
      returns a Pandas data frame containing the feature data. It may accept
      additional, optional arguments.

    - :meth:`fit`: Like Scikit-learn estimators, Apollo models must have a
      ``fit`` method for training the model. However, unlike Scikit-learn
      estimators, Apollo models have a single required argument, a target
      data frame to fit against. Additional arguments are forwarded to
      :meth:`load_data`.

    - :meth:`predict`: Again like Scikit-learn estimators, Apollo models must
      have a ``predict`` method for generating predictions. The input to this
      method is a Pandas index for the resulting prediction. The return value
      is a data frame of predictions. Additional arguments are forwarded to
      :meth:`load_data`.

    - :meth:`score`: All models have a score method, but unlike Scikit-learn,
      this method produces a data frame rather than a scalar value. The data
      frame has one column for each column in th training data, and each row
      gives a different metric. It is not specified what metrics should be
      computed nor how they should be interpreted. The default implementation
      delegates to :func:`apollo.metrics.all`.

    This base class provides default implementations of :meth:`fit` and
    :meth:`predict`. These defaults make use of the following lower-level
    pieces:

    - :attr:`estimator`: The default implementation wraps a Scikit-learn style
      estimator to perform the actual predictions. It recieves as input the
      values produced by :meth:`preprocess` (described below). You **must**
      provide an estimator attribute if you use the default :meth:`fit` or
      :meth:`predict` methods.

    - :meth:`preprocess`: This method transforms the feature data returned
      by :meth:`load_data` and target data provided by the user into numpy
      arrays to be passed to the estimator. This method may perform additional,
      possibly learned, preprocessing steps. Subclasses should call the base
      implementation to learn the column names from the targets.

    - :meth:`postprocess`: This method transforms the raw predictions
      returned by the estimator into a fully-fledged data frame. Subclasses
      should call the base implementation to ensure the resulting data frame
      has the proper column names.
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
                A Scikit-learn estimator or a string or list to be interpreted
                by :func:`apollo.models.make_estimator`.
        '''
        self._name = str(name or uuid.uuid4())
        self._estimator = make_estimator(estimator)
        self._columns = None

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
            index (apollo.DatetimeIndex):
                The index of the data.
            **kwargs:
                Implementations may accept additional, optional arguments.

        Returns:
            features (pandas.DataFrame):
                Feature data for the given index.
        '''
        pass

    def preprocess(self, features, targets=None):
        '''Convert feature and target data into numpy arrays for the estimator.

        The default implementation passes both the features and targets to
        :func:`numpy.asanyarray`. If targets is not ``None``, it records the
        column names from the targets.

        Arguments:
            features (pandas.DataFrame):
                The feature data returned by :meth:`load_data`.
            targets (pandas.DataFrame or None):
                The target data passed into :meth:`fit`. This argument will be
                a data frame during :meth:`fit` and ``None`` during
                :meth:`predict`.

        Returns:
            raw_features (numpy.ndarray):
                The features in a form that can be sent to the estimator.
            raw_targets (numpy.ndarray):
                The targets in a form that can be sent to the estimator. If
                no targets were given, this will be ``None``.
            index (apollo.DatetimeIndex):
                An index for the processed data.
        '''
        logger.debug('preprocess: casting to numpy')

        if targets is None:
            raw_features = np.asanyarray(features)
            raw_targets = None
            index = features.index

        else:
            raw_features = np.asanyarray(features)
            raw_targets = np.asanyarray(targets)
            index = features.index
            self._columns = list(targets.columns)

        return raw_features, raw_targets, index

    def postprocess(self, raw_predictions, index):
        '''Convert raw predictions into a :class:`pandas.DataFrame`.

        The default implementation simply delegates to the
        :class:`pandas.DataFrame` constructor.

        Arguments:
            raw_predictions (numpy.ndarray):
                The output of ``self.estimator.predict``.
            index (apollo.DatetimeIndex):
                The index of the resulting data frame.

        Returns:
            pandas.DataFrame:
                The predictions.
        '''
        logger.debug('postprocess: constructing data frame')
        return pd.DataFrame(raw_predictions, index=index, columns=self._columns)

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
        data, targets, _ = self.preprocess(data, targets)
        logger.debug('fit: fitting estimator')
        self.estimator.fit(data, targets)
        return self

    def predict(self, index, **kwargs):
        '''Generate a prediction from this model.

        Arguments:
            index (apollo.DatetimeIndex):
                Make predictions for this index.
            **kwargs:
                Additional arguments are forwarded to :meth:`load_data`.

        Returns:
            pandas.DataFrame:
                A data frame of predicted values.
        '''
        data = self.load_data(index, **kwargs)
        data, _, index = self.preprocess(data)
        logger.debug('predict: executing estimator')
        predictions = self.estimator.predict(data)
        predictions = self.postprocess(predictions, index)
        return predictions

    def score(self, targets, **kwargs):
        '''Score this model against some target values.

        Arguments:
            targets (pandas.DataFrame):
                The targets to compare against.
            **kwargs:
                Additional arguments are forwarded to :meth:`load_data`.

        Returns:
            pandas.DataFrame:
                A table of metrics.
        '''
        # Drop missing targets.
        targets = targets.replace([np.inf, -np.inf], np.nan).dropna()

        # Compute predictions.
        predictions = self.predict(targets.index, **kwargs)

        # Warn if we can't form a prediction for all targets.
        n_missing = len(targets.index) - len(predictions.index)
        if n_missing != 0:
            logger.warning(f'missing {n_missing} predictions')
            targets = targets.reindex(predictions.index)

        # Compute the scores.
        scores = metrics.all(targets, predictions)

        # Ensure the index column has a name.
        scores.index.name = 'metric'
        return scores

    def save(self, path=None):
        '''Persist a model to disk.

        This method simply delegates to :func:`apollo.models.write_model` or
        :func:`apollo.models.write_model_at` as appropriate.

        Arguments:
            path (str or pathlib.Path or None):
                The path at which to save the model. The default is a path
                within the Apollo database derived from the model's name.

        Returns:
            pathlib.Path:
                The path at which the model was saved.
        '''
        if path is None:
            return write_model(self)
        else:
            return write_model_at(self, path)


class IrradianceModel(Model):
    '''Base class for irradiance modeling.

    This class extends the :meth:`preprocess` and :meth:`postprocess` methods
    specifically for irradiance modeling. This class can:

    - Standardize features and targets for training.
    - Compute additional time-of-day and time-of-year features.
    - Discard training samples that occur when the sun is down.
    '''

    def __init__(
        self, *,
        standardize=False,
        add_time_of_day=True,
        add_time_of_year=True,
        daylight_only=False,
        latlon=None,
        **kwargs,
    ):
        '''Construct a new model.

        Keyword Arguments:
            name (str or None):
                A name for the estimator. If not given, a UUID is generated.
            estimator (sklearn.base.BaseEstimator or str or list):
                A Scikit-learn estimator or a string or list to be interpreted
                by :func:`apollo.models.make_estimator`.
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
            latlon (pair of float):
                The center of the geographic area, as a latitude-longited pair.
                Used to compute the sunrise and sunset times. Required only
                when ``daylight_only`` is True.
        '''
        super().__init__(**kwargs)

        self.standardize = bool(standardize)
        self.add_time_of_day = bool(add_time_of_day)
        self.add_time_of_year = bool(add_time_of_year)
        self.daylight_only = bool(daylight_only)
        self.latlon = latlon

        # The standardizers. The feature scaler may not be used.
        self.feature_scaler = StandardScaler(copy=False)
        self.target_scaler = StandardScaler(copy=False)

    def preprocess(self, data, targets=None):
        '''Convert feature and target data into numpy arrays for the estimator.

        Arguments:
            features (pandas.DataFrame):
                The feature data returned by :meth:`load_data`.
            targets (pandas.DataFrame or None):
                The target data passed into :meth:`fit`. This argument will be
                a data frame during :meth:`fit` and ``None`` during
                :meth:`predict`.

        Returns:
            raw_features (numpy.ndarray):
                The features in a form that can be sent to the estimator.
            raw_targets (numpy.ndarray):
                The targets in a form that can be sent to the estimator. If
                no targets were given, this will be ``None``.
            index (apollo.DatetimeIndex):
                An index for the processed data.
        '''
        # True when fitting the estimator.
        fitting = (targets is not None)

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
            (lat, lon) = self.latlon
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
        index = apollo.DatetimeIndex(index, name='time')

        # Scale the feature data (optionally).
        if self.standardize:
            logger.debug('preprocess: scaling features')
            cols = list(data.columns)
            raw_features = data[cols].to_numpy()
            if fitting: self.feature_scaler.fit(raw_features)
            data[cols] = self.feature_scaler.transform(raw_features)

        # Scale the target data (optionally).
        if self.standardize and targets is not None:
            logger.debug('preprocess: scaling targets')
            cols = list(targets.columns)
            raw_targets = targets[cols].to_numpy()
            if fitting: self.target_scaler.fit(raw_targets)
            targets[cols] = self.target_scaler.transform(raw_targets)

        # Compute additional features (optionally).
        if self.add_time_of_day:
            logger.debug('preprocess: computing time-of-day')
            data = data.join(apollo.time_of_day(index))
        if self.add_time_of_year:
            logger.debug('preprocess: computing time-of-year')
            data = data.join(apollo.time_of_year(index))

        # We must delegate the the base implementation.
        return super().preprocess(data, targets)

    def postprocess(self, raw_predictions, index):
        '''Convert raw predictions into a :class:`pandas.DataFrame`.

        Arguments:
            raw_predictions (numpy.ndarray):
                The output of ``self.estimator.predict``.
            index (apollo.DatetimeIndex):
                The index of the resulting data frame.

        Returns:
            pandas.DataFrame:
                The predictions.
        '''
        # Reconstruct the data frame.
        index = apollo.DatetimeIndex(index, name='time')
        predictions = super().postprocess(raw_predictions, index)

        # Unscale the predictions.
        if self.standardize:
            logger.debug('postprocess: unscaling predictions')
            cols = list(predictions.columns)
            predictions[cols] = self.target_scaler.inverse_transform(predictions)

        # Set overnight predictions to zero (optionally).
        if self.daylight_only:
            logger.debug('postprocess: setting night time to zero')
            (lat, lon) = self.latlon
            night = ~apollo.is_daylight(index, lat, lon)
            predictions.loc[night, :] = 0

        return predictions

    def score(self, targets, **kwargs):
        '''Score this model against some target values.

        Arguments:
            targets (pandas.DataFrame):
                The targets to compare against.
            **kwargs:
                Additional arguments are forwarded to :meth:`load_data`.

        Returns:
            pandas.DataFrame:
                A table of metrics.
        '''
        # Drop missing targets.
        targets = targets.replace([np.inf, -np.inf], np.nan).dropna()

        # Compute predictions.
        predictions = self.predict(targets.index, **kwargs)

        # Warn if we can't form a prediction for all targets.
        n_missing = len(targets.index) - len(predictions.index)
        if n_missing != 0:
            logger.warning(f'missing {n_missing} predictions')
            targets = targets.reindex(predictions.index)

        # Compute daytime mask.
        lat, lon = self.latlon
        is_daylight = apollo.is_daylight(targets.index, lat, lon)

        # Compute the scores.
        daynight_scores = metrics.all(targets, predictions)
        daynight_scores.index = daynight_scores.index + '_day_night'
        dayonly_scores = metrics.all(targets[is_daylight], predictions[is_daylight])
        dayonly_scores.index = dayonly_scores.index + '_day_only'
        scores = daynight_scores.append(dayonly_scores)

        # Ensure the index column has a name.
        scores.index.name = 'metric'
        return scores
