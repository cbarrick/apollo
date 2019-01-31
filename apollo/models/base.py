import abc
import json
import logging

from apollo import storage
from apollo.utils import get_concrete_subclasses


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

    # Find all subclasses of Model
    subclass_list = get_concrete_subclasses(Model)
    subclasses = {model.__name__: model for model in subclass_list}

    # Ensure there is a subclass of ``Model`` with the same name
    if cls_name not in subclasses:
        raise ValueError(
            f'Cannot find model class {cls_name}.\n'
            f'Available classes are {subclasses.keys()}.'
        )

    # Load the model
    cls = subclasses[cls_name]
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    model = cls.load(path)
    return model


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
