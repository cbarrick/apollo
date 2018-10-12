""" Abstract base class for models used to generate prediction files

Apollo Models are trainable predictors of solar irradiance.  Any object conforming to the Model API can be used with
the CLI found in apollo/__main__.py

"""

from abc import ABC, abstractmethod
import os
import json
import datetime
import pandas as pd


class Model(ABC):

    @abstractmethod
    def __init__(self, name):
        """ Initializes a Model

        Args:
            name (str):
                descriptive human-readable name for this model.  Typically a description of the regressor used such as
                "decision-tree" or "random forest".
        """
        super().__init__()
        self.name = name

    @abstractmethod
    def save(self, regressor, save_dir, target_hour, target_var):
        """ Serializes a trained regressor and saves it to disk

        Serialization/deserialization are methods because each Model might serialize regressors differently.
        For example, scikit-learn recommends using joblib to dump trained models, whereas most NN packages like
        TF and PyTorch have built-in serialization mechanisms.

        Args:
            regressor (object):
                the object to serialize.
            save_dir (str):
                directory where the model should be serialized.
            target_hour (int):
                future hour targeted by the model.
            target_var (str):
                name of the variable targeted by the model.

        Returns:
            str: location of the serialized model.

        """
        pass

    @abstractmethod
    def load(self, save_dir, target_hour, target_var):
        """ Deserializes a regressor from disk

        Args:
            save_dir (str):
                directory where the serialized model is saved.
            target_hour (int):
                future hour targeted by the model.
            target_var (str):
                name of the variable targeted by the model.

        Returns:
            object: deserialized regressor if a saved regressor is found.  Otherwise, returns None.

        """
        pass

    @abstractmethod
    def train(self, begin_date, end_date, target_hour, target_var, save_dir, tune, num_folds):
        """ Trains the model to predict <target_var> at time <target_hour> in the future

        This method will construct a SolarDataset spanning the time from begin_date to end_date and use the dataset
        to train the model.

        Args:
            begin_date (str):
                timestamp corresponding to the first training instance.
            end_date (str):
                timestamp corresponding to the last training instance.
            target_hour (int):
                future hour to target.
            target_var (str):
                name of variable to target.
            save_dir (str):
                directory where the trained model should be serialized.
            tune (bool):
                if true, cross-validated parameter tuning will be done.
            num_folds (int):
                number of folds to use for cross-validated parameter tuning.

        Returns:
            str: filepath where trained model is serialized.

        """
        pass

    @abstractmethod
    def evaluate(self, begin_date, end_date, target_hour, target_var, save_dir, num_folds, metrics):
        """ Evaluate a trained model using cross validation

        Args:
            begin_date (str):
                timestamp corresponding to the first reftime to use for validation.
            end_date (str):
                timestamp corresponding to the last reftime to use for validation.
            target_hour (int):
                future hour to target.
            target_var (str):
                name of variable to target.
            save_dir (str):
                directory where the trained model is currently serialized.
            num_folds (int):
                number of folds to use.
            metrics (dict):
                metrics that should be used for evaluation.  The key should be the metric's name, and the
                value should be a function that implements that metric.

        Returns:
            dict: mapping of metrics to results. Dictionary entries map metric names to evaluation results.
        """
        pass

    @abstractmethod
    def predict(self, begin_date, end_date, target_hour, target_var, save_dir):
        """ Predict future solar irradiance readings using a trained model

        Predictions are output as two json files: a summary file and a prediction file.
        The summary file contains metadata about the predictions.  The prediction file contains column metadata and
        the raw predictions.

        Args:
            begin_date (str):
                timestamp of the first reference time.
            end_date (str):
                timestamp of the last reference time.
            target_hour (int):
                future hour to target.
            target_var (str):
                name of variable to target.
            save_dir (str):
                directory where the trained model is currently serialized.

        Returns:
            numpy.array: array of [reftime, prediction] pairs.  Array shape should be n x 2

        """
        pass

    def write_predictions(self, predictions, begin_date, end_date, target_hour, target_var, summary_dir, output_dir):
        """ Write the predictions generated by the model to a file

        Two files are generated - a summary file and a prediction file.
        The summary file provides meta-data on the predictions made by a model, including the model name, date created,
        start/end dates, and location of the prediction file.  The prediction file contains column metadata and
        the raw predictions.

        Args:
            predictions (numpy.array):
                n x 2 numpy array containing the model's predictions.
                The first column should be the timestamp (str) of the reference time.  The second column should be the
                predicted solar irradiance.
            begin_date (str):
                timestamp of the reftime of the first prediction.
            end_date (str):
                timestamp of the reftime of the last prediction.
            target_hour (int):
                future hour of the predictions
            target_var (str):
                name of the target variable
            summary_dir (str or path):
                directory where the summary file should be written
            output_dir (str):
                directory where the prediction file should be written

        Returns:
            (str, str): path to summary file, path to prediction file

        """
        # convert string timestamps to posix time
        formatted_predictions = list(map(
            lambda prediction: [Model._datestring_to_posix(prediction[0]), prediction[1]],
            predictions
        ))

        # ensure output directories exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        # create path to summary and to resource files
        model_name = self._generate_name(target_var, target_hour)
        summary_filename = f'{model_name}_{begin_date}_{end_date}.summary.json'
        summary_path = os.path.join(summary_dir, summary_filename)
        summary_path = os.path.realpath(summary_path)

        resource_filename = f'{model_name}_{begin_date}_{end_date}.json'
        resource_path = os.path.join(output_dir, resource_filename)
        resource_path = os.path.realpath(resource_path)

        summary_dict = {
            'source': self.name,
            'sourcelabel': self.name.replace('_', ' '),
            'site': target_var,
            'created': round(datetime.datetime.utcnow().timestamp()),
            'start': Model._datestring_to_posix(begin_date),
            'stop': Model._datestring_to_posix(end_date),
            'resource': resource_path
        }

        data_dict = {
            'start': Model._datestring_to_posix(begin_date),
            'stop': Model._datestring_to_posix(end_date),
            'site': target_var,
            'columns': [
                {
                    'label': 'TIMESTAMP',
                    'units': '',
                    'longname': '',
                    'type': 'datetime'
                },
                {
                    'label': target_var,
                    'units': 'w/m2',
                    'longname': '',
                    'type': 'number'
                },
            ],
            'rows': formatted_predictions
        }

        # write the summary file
        with open(summary_path, 'w') as summary_file:
            json.dump(summary_dict, summary_file, separators=(',', ':'))

        # write the file containing the data
        with open(resource_path, 'w') as resource_file:
            json.dump(data_dict, resource_file, separators=(',', ':'))

        return summary_path, resource_path

    def _generate_name(self, target_hour, target_var):
        """ Create a unique name for a model that predicts a named target variable at a specific future hour

        Args:
            target_hour (int):
                The future hour targeted by this model.
            target_var (str):
                Name of the target variable.

        Returns:
            str: unique name for the model.

        """
        return f'{self.name}_{target_hour}hr_{target_var}.model'

    @classmethod
    def _datestring_to_posix(cls, date_string):
        timestring = pd.to_datetime(date_string, utc=True).timestamp()
        return round(timestring) * 1000  # convert to milliseconds
