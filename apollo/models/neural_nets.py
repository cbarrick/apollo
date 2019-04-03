from sklearn.neural_network import MLPRegressor

from apollo.models.scikit_estimator import ScikitModel


class MultilayerPerceptron(ScikitModel):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._estimator = MLPRegressor()

    @property
    def estimator(self):
        return self._estimator

    @property
    def default_hyperparams(self):
        return {
            'hidden_layer_sizes': (57, 24,),
            'activation': 'relu',
            'solver': 'adam',
            'batch_size': 100,
            'max_iter': 500,
        }