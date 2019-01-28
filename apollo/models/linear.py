from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

from apollo.models.scikit_estimator import ScikitModel


class LinReg(ScikitModel):
    @property
    def estimator(self):
        return LinearRegression()

    @property
    def default_hyperparams(self):
        return {}


class RandomForest(ScikitModel):
    @property
    def estimator(self):
        return SVR()

    @property
    def default_hyperparams(self):
        return {
            'C': 1.4,
            'epsilon': 0.6,
            'kernel': 'sigmoid',
            'gamma': 0.001
        }
