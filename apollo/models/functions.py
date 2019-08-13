import json
import logging
from pathlib import Path

import pickle5 as pickle

import sklearn
from sklearn.pipeline import make_pipeline

import apollo


logger = logging.getLogger(__name__)


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
            if isinstance(name, str):
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


def write_model_to(model, stream):
    '''Serialize a model to a binary stream.

    Arguments:
        model (apollo.models.base.Model):
            The model to serialize.
        stream (io.IOBase):
            A writable, binary stream.
    '''
    pickle.dump(model, stream, protocol=5)


def write_model_at(model, path):
    '''Persist a model to disk.

    Arguments:
        model (apollo.models.base.Model):
            The model to serialize.
        path (str or pathlib.Path or None):
            The path at which to save the model. The default is a path
            within the Apollo database derived from the model's name.

    Returns:
        pathlib.Path:
            The path at which the model was saved.
    '''
    logger.debug(f'save: writing model to {path}')
    path = Path(path).resolve()
    stream = path.open('wb')
    write_model_to(model, stream)
    return path


def write_model(model):
    '''Persist a model to the Apollo database.

    Arguments:
        model (apollo.models.base.Model):
            The model to serialize.

    Returns:
        pathlib.Path:
            The path at which the model was saved.
    '''
    path = apollo.path(f'models/{model.name}.model')
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_model_at(model, path)
