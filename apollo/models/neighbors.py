from sklearn.neighbors import KNeighborsRegressor

from apollo.models.scikit_estimator import ScikitModel


class KNearest(ScikitModel):
    @property
    def estimator(self):
        return KNeighborsRegressor()

    @property
    def default_hyperparams(self):
        return {
            'n_neighbors': 5,
            'weights': 'distance',
        }
