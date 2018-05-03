"""
Abstract base class for experiments
"""

_CACHE_DIR = '../data'  # where the NAM and GA-POWER data resides
_MODELS_DIR = '../models'  # directory where serialized models will be saved
_OUTPUT_DIR = '../predictions'  # directory where predictions are saved
_DEFAULT_TARGET = 'UGA-C-POA-1-IRR'


class Experiment(object):

    def __init__(self, name):
        self.name = name

    def make_model_name(self, target_hour, target_var):
        # creates a unique name for a model that predicts a specific target variable at a specific target hour
        return f'{self.name}_{target_hour}hr_{target_var}.model'

    def save(self, model, save_dir, target_hour, target_var):
        pass

    def load(self, save_dir, target_hour, target_var):
        pass

    def train(self, begin_date='2017-12-01 00:00', end_date='2017-12-31 18:00', target_hour=24, target_var=_DEFAULT_TARGET,
              cache_dir=_CACHE_DIR, save_dir=_MODELS_DIR, tune=True, num_folds=3):
        pass

    def evaluate(self, begin_date='2017-12-01 00:00', end_date='2017-12-31 18:00', target_hour=24, target_var=_DEFAULT_TARGET,
                 cache_dir=_CACHE_DIR, save_dir=_MODELS_DIR, metrics=['neg_mean_absolute_error'], num_folds=3):
        pass

    def predict(self, begin_date, end_date, target_hour=24, target_var=_DEFAULT_TARGET,
                cache_dir=_CACHE_DIR, save_dir=_MODELS_DIR, output_dir=_OUTPUT_DIR):
        pass

