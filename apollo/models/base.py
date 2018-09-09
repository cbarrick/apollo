""" Abstract base class for models used to generate prediction files

Apollo Models are trainable predictors of solar irradiance.  Any object conforming to the Model API can be used with
the CLI found in apollo/__main__.py

"""

from abc import ABC, abstractmethod


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
    def train(self, begin_date, end_date, target_hour, target_var, cache_dir, save_dir, tune, num_folds):
        """ Trains the model to predict <target_var> at time <target_hour> in the future

        This method will construct a SolarDataset spanning the time from begin_date to end_date and use the dataset
        to train the model.

        Args:
            begin_date (str):
                datetime string corresponding to the first training instance.
            end_date (str):
                datetime string corresponding to the last training instance.
            target_hour (int):
                future hour to target.
            target_var (str):
                name of variable to target.
            cache_dir (str):
                directory where training data is saved.
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
    def evaluate(self, begin_date, end_date, target_hour, target_var, cache_dir, save_dir, num_folds, metrics):
        """ Evaluate a trained model using cross validation

        Args:
            begin_date (str):
                datetime string corresponding to the first training instance.
            end_date (str):
                datetime string corresponding to the last training instance.
            target_hour (int):
                future hour to target.
            target_var (str):
                name of variable to target.
            cache_dir (str):
                directory where training data is saved.
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
    def predict(self, begin_date, end_date, target_hour, target_var, cache_dir, save_dir, summary_dir, output_dir):
        """ Predict future solar irradiance readings using a trained model

        TODO: we should separate the method that generate predictions from the method that writes the output files

        Predictions are output as two json files: a summary file and a prediction file.
        The summary file contains metadata about the predictions.  The prediction file contains column metadata and
        the raw predictions.

        Args:
            begin_date (str):
                datetime string corresponding to the first training instance.
            end_date (str):
                datetime string corresponding to the last training instance.
            target_hour (int):
                future hour to target.
            target_var (str):
                name of variable to target.
            cache_dir (str):
                directory where training data is saved.
            save_dir (str):
                directory where the trained model is currently serialized.
            summary_dir (str):
                directory where summary file should be written.
            output_dir:
                directory where prediction file should be written.

        Returns:
            (str, str): path to the summary file, path to the prediction file.
        """
        pass

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
