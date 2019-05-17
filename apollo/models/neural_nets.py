from sklearn.neural_network import MLPRegressor

from apollo.models.scikit_estimator import ScikitModel


class MultilayerPerceptron(ScikitModel):
    ''' Feed-forward neural network

    Multi-layer Perceptrons are fully-connected, feed-forward neural networks
    consisting of an input layer, one or more hidden layers,
    and an output layer.  Each node in the network uses a non-linear activation
    function and the weights between nodes are trained using the
    backpropogation algorithm.
    Neural networks are capable of modeling complex non-linear functions, but
    often require careful training, tuning, and regularization to perform well.

    This model accepts the following hyperparameters, which can be set by
    passing kwargs with matching names to the model's constructor:

    hidden_layer_sizes: Tuple[int, int, ... , int] (default ``(57, 24)``)
        Defines the shape of the network.  The `i`th value of the parameter
        is the number of nodes in the `i`th hidden layer.

        For example, ``hidden_layer_sizes = (10, 5)`` results in a network with
        an input layer, 10 nodes in the first hidden layer, 5 nodes in the
        second hidden layer, and an output layer.

    activation: string (default 'relu')
        Activation function.  Should be one of ‘identity’, ‘logistic’, ‘tanh’,
        or ‘relu’.

        - 'identity' is a no-op activation.
        - 'logistic' is :math:`\\frac{1}{1 + e^{-x}}`.
        - 'tanh' is :math:`tanh(x)`
        - 'relu' returns ``max(0, x)``

    solver: string (default 'adam')
        Solver for weight optimization.  Should be one of 'sgd', 'lbfgs',
        or 'adam'.

        - 'sgd': Stochastic Gradient Descent.
        - 'lbfgs': "Limited-memory Broyden–Fletcher–Goldfarb–Shanno"
        algorithm.  At a high level, this solver is a second-order method that
        may converge more quickly than sgd.  However, it takes much more memory
        and computation.
        - 'adam': ADaptive Moment Estimation algorithm.
        This is an extension of stochastic gradient descent that automatically
        adapts the learning rate and momentum parameters.

    batch\_size: int (default 100)
        Size of mini-batches to use for 'sgd' or 'adam' solvers.

    max\_iter: int (default 500)
        Maximum number of passes to make through the training set.

        The actual number of passes may be different is convergence is detected.
    '''
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