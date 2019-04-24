from sklearn.svm import SVR as scikit_SVR
from sklearn.linear_model import LinearRegression as scikit_LinearRegression

from apollo.models.scikit_estimator import ScikitModel


class LinearRegression(ScikitModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._estimator = scikit_LinearRegression()

    @property
    def estimator(self):
        return self._estimator

    @property
    def default_hyperparams(self):
        return {}


class SVR(ScikitModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._estimator = scikit_SVR()

    @property
    def estimator(self):
        return self._estimator

    @property
    def default_hyperparams(self):
        return {
            'C': 1000,
            'epsilon': 2,
            'kernel': 'rbf',
            'gamma': 0.0001
        }
