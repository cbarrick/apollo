from sklearn.neighbors import KNeighborsRegressor

from apollo.models.scikit_estimator import ScikitModel


class KNearest(ScikitModel):
    ''' k-Nearest Neighbors regressor

    KNN is an instance-based model which predicts the target variable using a
    local interpolation among the `k` examples from the training set
    that are most similar to the instance being predicted.
    Instance-based models are cheap to train, but their accuracy depends on the
    quality and quantity of data collected, and they require additional memory
    to store the data structure created during training.

    This model accepts the following hyperparameters, which can be set by
    passing kwargs with matching names to the model's constructor:

    n_neighbors: int (default 5)
        The number of neighbors to consider.
        Too-high values will increase bias and may cause the model to underfit.
        Too-low values will increase variance and may cause the model to
        overfit.

    weights: string or Callable (default 'distance')
        Function used to weight neighbors.
        If a string, should be one of 'uniform' or 'distance'.

        If uniform weighting is used, then each neighbor will be given equal
        weight when computing the interpolated value.
        Distance-based weighting assigns weights to each neighbor based on the
        inverse of the distance from the instance under consideration.
        Similar neighbors will contribute more to the predicted value than
        dissimilar neighbors.

    p: int (default 2)
        Order of the distance metric.
        The Apollo KNN uses the Minkowski distance metric, which is defined
        between vectors :math:`X = <x_1, x_2, ... x_n>`
        and `Y = <y_1, y_2, ... y_n>` as
        :math:`D(X, Y) = (\\sum_{i=1}^{n} |{x_i - y_i}|^p)^{\\frac{1}{p}}`.

        With `p` = 1, this metric is the Manhattan distance metric.
        With `p` = 2, this metric is the familiar Euclidean distance metric.
    '''
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._estimator = KNeighborsRegressor()

    @property
    def estimator(self):
        return self._estimator

    @property
    def default_hyperparams(self):
        return {
            'n_neighbors': 6,
            'weights': 'distance',
            'p': 2
        }
