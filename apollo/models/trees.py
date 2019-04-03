from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from apollo.models.scikit_estimator import ScikitModel


class DecisionTree(ScikitModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._estimator = DecisionTreeRegressor()

    @property
    def estimator(self):
        return self._estimator

    @property
    def default_hyperparams(self):
        return {
            'splitter': 'best',
            'max_depth': 20,
            'min_impurity_decrease': 0.25
        }


class RandomForest(ScikitModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._estimator = RandomForestRegressor()

    @property
    def estimator(self):
        return self._estimator

    @property
    def default_hyperparams(self):
        return {
            'n_estimators': 100,
            'max_depth': 50,
            'min_impurity_decrease': 0.30
        }


class GradientBoostedTrees(ScikitModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._estimator = XGBRegressor()

    @property
    def estimator(self):
        return self._estimator

    @property
    def default_hyperparams(self):
        return {
            'learning_rate': 0.05,
            'n_estimators': 200,
            'max_depth': 20,
        }