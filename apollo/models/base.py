""" Abstract base class for models used to generate prediction files

Apollo Models are trainable predictors of solar irradiance.  Any object conforming to the Model API can be used with
the CLI found in apollo/__main__.py

"""

_CACHE_DIR = '../data'  # where the NAM and GA-POWER data resides
_MODELS_DIR = '../models'  # directory where serialized models will be saved
_OUTPUT_DIR = '../predictions'  # directory where predictions are saved
_DEFAULT_TARGET = 'UGA-C-POA-1-IRR'


class Model(object):

    def __init__(self, name):
        self.name = name

    def make_model_name(self, target_hour, target_var):
        # creates a unique name for a model that predicts a specific target variable at a specific target hour
        return f'{self.name}_{target_hour}hr_{target_var}.model'

    def save(self, model, save_dir, target_hour, target_var):
        pass

    def load(self, save_dir, target_hour, target_var):
        pass

    def train(self, begin_date, end_date, target_hour, target_var, cache_dir, save_dir, tune, num_folds):
        pass

    def evaluate(self, begin_date, end_date, target_hour, target_var, cache_dir, save_dir, num_folds):
        pass

    def predict(self, begin_date, end_date, target_hour, target_var, cache_dir, save_dir, summary_dir, output_dir):
        pass

