from sklearn.neural_network import MLPRegressor

from apollo.models.scikit_estimator import ScikitModel


class MultilayerPerceptron(ScikitModel):
    @property
    def estimator(self):
        return MLPRegressor()

    @property
    def default_hyperparams(self):
        return {
            'hidden_layer_sizes': (57, 24,),
            'activation': 'relu',
            'solver': 'adam',
            'batch_size': 100,
            'max_iter': 500,
        }