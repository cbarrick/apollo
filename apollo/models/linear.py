from sklearn.svm import SVR as scikit_SVR
from sklearn.linear_model import LinearRegression as scikit_LinearRegression

from apollo.models.scikit_estimator import ScikitModel


class LinearRegression(ScikitModel):
    @property
    def estimator(self):
        return scikit_LinearRegression()

    @property
    def default_hyperparams(self):
        return {}


class SVR(ScikitModel):
    @property
    def estimator(self):
        return scikit_SVR()

    @property
    def default_hyperparams(self):
        return {
            'C': 1.4,
            'epsilon': 0.6,
            'kernel': 'rbf',
            'gamma': 0.001
        }
