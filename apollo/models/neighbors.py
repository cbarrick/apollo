from sklearn.neighbors import KNeighborsRegressor

from apollo.models.scikit_estimator import ScikitModel


class KNearest(ScikitModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._estimator = KNeighborsRegressor()

    @property
    def estimator(self):
        return self._estimator

    @property
    def default_hyperparams(self):
        return {
            'n_neighbors': 5,
            'weights': 'distance',
        }
