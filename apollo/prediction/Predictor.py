""" Abstract base class for models that produce solar radiation predictions

Any object conforming to the Predictor API can be used with the prediction CLI (apollo/prediction/__main__.py)

"""

from abc import ABC, abstractmethod
import os
import json
import datetime
import pandas as pd


class Predictor(ABC):

    @abstractmethod
    def __init__(self, name, target, target_hours):
        """ Interface for predictors of solar radiation

        Args:
            name (str):
                A descriptive human-readable name for this predictor.
                Typically the type of the regressor used such as "decision-tree" or "random forest".

            target (str):
                The name of the variable to target

            target_hours (Iterable[int]):
                The future hours to be predicted.
        """
        super().__init__()
        self.name = name
        self.target_hours = target_hours
        self.target = target
        self.filename = f'{self.name}_{target_hours[0]}hr-{target_hours[-1]}hr_{target}.model'

    @abstractmethod
    def save(self, save_dir):
        """ Serializes this regressor backing this predictor to a file

        Serialization/deserialization are abstract methods because each Predictor might serialize regressors
        differently.
        For example, scikit-learn recommends using joblib to dump trained regressors, whereas most NN packages like
        TF and PyTorch have built-in serialization mechanisms.

        Args:
            save_dir (str):
                Directory where the regressor should be saved.

        Returns:
            str: location of the serialized regressor.

        """
        pass

    @abstractmethod
    def load(self, save_dir):
        """ Deserializes a regressor from a file

        Args:
            save_dir (str):
                The directory where the serialized regressor is saved.

        Returns:
            object or None: deserialized regressor if a saved regressor is found.  Otherwise, None.

        """
        pass

    @abstractmethod
    def train(self, start, stop, save_dir, tune, num_folds):
        """ Fits the predictor and saves it to disk

        Trains the predictor to predict `self.target` at each future hour in `self.target_hours` using
        a `SolarDataset` with reftimes between `start` and `stop`.

        Args:
            start (str):
                Timestamp corresponding to the reftime of the first training instance.
            stop (str):
                Timestamp corresponding to the reftime of the final training instance.
            save_dir (str):
                The directory where the trained model should be saved.
            tune (bool):
                If true, perform cross-validated parameter tuning before training.
            num_folds (int):
                The number of folds to use for cross-validated parameter tuning.  Ignored if `tune` == False.

        Returns:
            str: The path to the serialized predictor.

        """
        pass

    @abstractmethod
    def cross_validate(self, start, stop, save_dir, num_folds, metrics):
        """ Evaluate this predictor using cross validation

        Args:
            start (str):
                Timestamp corresponding to the first reftime of the validation set.
            stop (str):
                Timestamp corresponding to the final reftime of the validation set.
            save_dir (str):
                The directory where the trained predictor is saved.
            num_folds (int):
                The number of folds to use.
            metrics (dict):
                Mapping of metrics that should be used for evaluation.  The key should be the metric's name, and the
                value should be a scoring function that implements the metric.

        Returns:
            dict: mapping of metric names to scores.
        """
        pass

    @abstractmethod
    def predict(self, start, stop, save_dir):
        """ Predict future solar irradiance values

        Predictions are output as two json files: a summary file and a prediction file.
        The summary file contains metadata about the predictions.  The prediction file contains column metadata and
        the raw predictions.

        Args:
            start (str):
                Timestamp of the first reference time to predict.
            stop (str):
                Timestamp of the final reference time to predict.
            save_dir (str):
                The directory where the trained predictor is saved.

        Returns:
            numpy.array: n x 2 array of [reftime, prediction] pairs.

        """
        pass

    def write_predictions(self, predictions, summary_dir, output_dir):
        """ Write the predictions generated by the model to a file

        Two files are generated - a summary file and a prediction file.
        The summary file provides meta-data on the predictions made by a model, including the model name, date created,
        start/end dates, and location of the prediction file.  The prediction file contains column metadata and
        the raw predictions.

        TODO: consider adopting a conventional path for summary and output directories

        Args:
            predictions (numpy.array):
                An n x 2 array of [reftime, predicted_value] pairs.
            summary_dir (str or path):
                The directory where the summary file should be written
            output_dir (str):
                The directory where the prediction file should be written

        Returns:
            (str, str): Path to summary file, Path to prediction file

        """

        # TODO: this is very broken since predictors now predict a window of future hours

        # convert string timestamps to posix time
        formatted_predictions = list(map(
            lambda prediction: [Predictor._datestring_to_posix(prediction[0]), prediction[1]],
            predictions
        ))

        # ensure output directories exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        # create path to summary and to resource files
        start_date, stop_date = predictions[0][0], predictions[-1][0]  # Note: assumes predictions are sorted
        summary_filename = f'{self.filename}_{start_date}_{stop_date}.summary.json'
        summary_path = os.path.join(summary_dir, summary_filename)
        summary_path = os.path.realpath(summary_path)

        resource_filename = f'{self.filename}_{start_date}_{stop_date}.json'
        resource_path = os.path.join(output_dir, resource_filename)
        resource_path = os.path.realpath(resource_path)

        summary_dict = {
            'source': self.name,
            'sourcelabel': self.name.replace('_', ' '),
            'site': self.target,
            'created': round(datetime.datetime.utcnow().timestamp()),
            'start': Predictor._datestring_to_posix(start_date),
            'stop': Predictor._datestring_to_posix(stop_date),
            'resource': resource_path
        }

        data_dict = {
            'start': Predictor._datestring_to_posix(start_date),
            'stop': Predictor._datestring_to_posix(stop_date),
            'site': self.target,
            'columns': [
                {
                    'label': 'TIMESTAMP',
                    'units': '',
                    'longname': '',
                    'type': 'datetime'
                },
                {
                    'label': self.target,
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

    @classmethod
    def _datestring_to_posix(cls, date_string):
        timestring = pd.to_datetime(date_string, utc=True).timestamp()
        return round(timestring) * 1000  # convert to milliseconds
