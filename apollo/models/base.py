import abc
import json
import logging
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

from apollo import storage
import apollo.datasets.ga_power as ga_power
from apollo.datasets.nam import CacheMiss
import apollo.models  # makes model subclasses discoverable


logger = logging.getLogger(__name__)


class Model(abc.ABC):
    '''The abstract base class for all models.
    '''

    @property
    @abc.abstractmethod
    def name(self):
        '''A string identifier for the model instance.
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, path):
        '''Load an instance of the model from a path.

        Arguments:
            path (pathlib.Path):
                The directory in which the model is stored.

        Returns:
            Model:
                An instance of the model.
        '''
        pass

    @abc.abstractmethod
    def save(self, path):
        '''Save the model to a path.

        Arguments:
            path (pathlib.Path):
                The directory in which to store model.
        '''
        pass

    def delete(self):
        delete(self.name)

    @abc.abstractmethod
    def fit(self, first, last):
        '''(Re)Train this model.

        The model is responsible for loading its own data.

        The reftimes of the training data should range from ``first`` to
        ``last``, inclusive.

        Arguments:
            first (timestamp):
                The timestamp of the first data point in the train set.
            last (timestamp):
                The timestamp of the last data point in the train set.
        '''
        pass

    @abc.abstractmethod
    def forecast(self, reftime):
        '''Make a forecast for the given reftime.

        The model is responsible for loading its own data.

        Arguments:
            reftime (pandas.Timestamp):
                The reference time of the forecast.

        Returns:
            forecast (pandas.Dataframe):
                A dataframe of forecasts with a :class:`~pandas.DatetimeIndex`.
                The index gives the timestamp of the forecast hours, and each
                column corresponds to a target variable being forecast.
        '''
        pass

    @property
    @abc.abstractmethod
    def target(self):
        ''' The name of the variable that this model targets.

        This is used with one of the loaders in apollo.datasets.ga_power to load
        true values of predicted solar irradiance.  It should match the name of
        one of the variables in the ga_power dataset.

        Returns:
            str: name of the target variable.

        '''
        pass

    @property
    @abc.abstractmethod
    def target_hours(self):
        ''' Hours for which this model makes predictions

        Returns:
            tuple: hours targeted by this model.
        '''
        pass

    def validate(self, first, last, splitter=TimeSeriesSplit(n_splits=5),
                 metrics=(mean_absolute_error,), **kwargs):
        ''' Estimate the accuracy of the model

        Args:
            first (str or Timestamp):
                The first reftime in the testing set.
            last (str or Timestamp):
                The last reftime in the testing set.
            splitter (object):
                An object implementing a `split` method, that returns training
                and testing indicies.
            metrics (Iterable[Callable]):
                Set of evaluation metrics to apply.  Each metric should have
                a signature `metric_name(y_true, y_predicted)`.
            **kwargs:
                Additional parameters to be passed to each metric.

        Returns:
            dict:
                A mapping from the name of each metric to the estimated error(s)
                computed by that metric.

        '''
        # find all reftimes in the testing set
        first = pd.Timestamp(first).floor(freq='6h')
        last = pd.Timestamp(last).floor(freq='6h')
        reftimes = pd.date_range(first, last, freq='6h')

        max_target_hour = max(self.target_hours)
        targets_last = last + pd.Timedelta(max_target_hour+1, 'h')

        # pre-load all ground truth readings
        ground_truth = ga_power.open_sqlite(
            self.target,
            start=first,
            stop=targets_last).to_dataframe()

        ground_truth.rename(
            columns={ground_truth.columns[0]: 'true_val'},
            inplace=True)

        evaluations = {metric.__name__: [] for metric in metrics}
        for train_index, test_index in splitter.split(reftimes):
            train_reftimes = reftimes[train_index]
            test_reftimes = reftimes[test_index]

            # train the model using the training set
            self.fit(train_reftimes[0], train_reftimes[-1])

            y_true, y_pred = [], []
            # make predictions for each reftime in the testing set
            for reftime in test_reftimes:
                try:
                    predictions = self.forecast(reftime)
                    # predictions will be a DataFrame
                    # of (timestamp, target) pairs for each target hour
                    predictions.rename(
                        columns={predictions.columns[0]: 'predicted'},
                        inplace=True)

                    # match predictions with ground truth
                    matched = pd.concat([predictions, ground_truth],
                                        axis=1, join='inner')

                    true_vals = matched['true_val'].values
                    pred_vals = matched['predicted'].values
                    assert(len(true_vals) == len(self.target_hours))
                    assert(len(pred_vals) == len(self.target_hours))
                    y_true.append(true_vals)
                    y_pred.append(pred_vals)

                # if data unavailable, omit the results from error estimation
                except CacheMiss:
                    logger.warning(f'Omitting results for reftime {reftime}')
                    pass

                # if some of the target hours were missing
                except AssertionError:
                    logger.warning(f'Omitting results for reftime {reftime}')
                    pass

            # compute error metrics for this split
            for metric in metrics:
                error = metric(y_true, y_pred, **kwargs)
                evaluations[metric.__name__].append(error)

        # find mean errors across all splits
        metrics = {m: np.mean(np.asarray(evaluations[m]), axis=0)
                   for m in evaluations}
        return metrics


def save(model):
    '''Save a model to the managed storage.

    This will overwrite any previously saved model with the same name.

    Arguments:
        model (Model):
            The model to be saved.
    '''
    root = storage.get('models')

    # The model must be a subclass of ``Model``.
    if not isinstance(model, Model):
        raise TypeError(
            f'Expected model to be a subclass of Model.\n'
            f'Got class {model.__class__.__name__}.\n'
        )

    # Save a mapping from model name to class name in ``manifest.json``.
    manifest = root / 'manifest.json'
    if not manifest.exists(): manifest.write_text('{}')
    manifest_text = manifest.read_text()
    manifest_json = json.loads(manifest_text)
    manifest_json[model.name] = model.__class__.__name__
    manifest_text = json.dumps(manifest_json)
    manifest.write_text(manifest_text)

    # Save the model under a directory with a matching name.
    path = root / model.name
    path.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load(name):
    '''Load a model from managed storage given its name.

    The model must have previously been saved with `save`.

    Notes:
        This function searches the subclass DAG of :class:`Model` for the
        proper class to load the model. If the class has not been imported, it
        will not be found.

    Arguments:
        name (str):
            The name of a previously saved model.

    Returns:
        Model:
            The model.
    '''
    root = storage.get('models')

    # Read ``manifest.json`` to figure out which class to use.
    manifest = root / 'manifest.json'
    if not manifest.exists(): manifest.write_text('{}')
    manifest_text = manifest.read_text()
    manifest_json = json.loads(manifest_text)
    cls_name = manifest_json.get(name)

    if cls_name is None:
        raise ValueError(f'Cannot find saved model with name `{name}`.')

    path = root / name

    # Search for a subclass of ``Model`` with the same name.
    for cls in list_known_models():
        if cls.__name__ == cls_name:
            model = cls.load(path)
            return model
    else:
        raise ValueError(f'Cannot find model of type {cls_name}.\n')


def delete(name):
    ''' Remove a model from managed storage given its name.

    The model must have previously been saved with `save`.

    Args:
        name (str):
            The name of a previously saved model

    Returns:
        None
    '''
    root = storage.get('models')

    # Read ``manifest.json`` to figure out which class to use.
    manifest_path = root / 'manifest.json'
    if not manifest_path.exists(): manifest_path.write_text('{}')
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    if name in manifest:
        # remove the files saved by the model
        model_dir = root / name
        shutil.rmtree(model_dir)

        # remove the entry from the manifest
        del manifest[name]
        with open(manifest_path, 'w') as manifest_file:
            json.dump(manifest, manifest_file)
    else:
        raise ValueError(f'Cannot find saved model with name `{name}`.')


def list_known_models():
    ''' Lists the subclasses of Model

    Returns:
        list of cls:
            List of Model classes
    '''
    subclasses = Model.__subclasses__()
    for cls in subclasses:
        subclasses.extend(cls.__subclasses__())

    return subclasses


def list_trained_models():
    ''' Lists the names of models which have been saved to the manifest file

    Returns:
        list of str:
            List of the names of models found in the manifest
    '''
    root = storage.get('models')
    manifest = root / 'manifest.json'
    if not manifest.exists():
        return []
    else:
        manifest_text = manifest.read_text()
        manifest_json = json.loads(manifest_text)
        return list(manifest_json.keys())
