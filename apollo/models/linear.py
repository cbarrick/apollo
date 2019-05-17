from sklearn.svm import SVR as scikit_SVR
from sklearn.linear_model import LinearRegression as scikit_LinearRegression

from apollo.models.scikit_estimator import ScikitModel


class LinearRegression(ScikitModel):
    ''' Least-square multiple linear regression

    This model attempts to predict the target variable by computing a
    weighted sum of the model inputs (plus an intercept term).
    The weights are determined using the "least-squares" approach, where
    weights are chosen analytically to minimize the sum of residuals.

    This model does not accept any hyperparameters.  All kwargs are forwarded
    to the data loader.
    '''
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
    ''' Support-Vector Regression

    Support-Vector machines (SVMs) attempt to find a *maximum-margin hyperplane*
    that predicts the target variable using the fewest possible examples from
    the dataset used for training.
    This typically makes support-vector models resistant to overfitting.

    This model accepts the following hyperparameters, which can be set by
    passing kwargs with matching names to the model's constructor:

    kernel: string or Callable (default 'rbf')
        The type of kernel to be used.
        It should be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, or Callable.

        The kernel is a function mapping the input to a different feature space.
        Using a linear kernel with an SVM is equivalent to building a
        maximum-margin linear regression.
        Using a non-linear kernel allows the SVM to model non-linear relationships.

    epsilon: float (default 0.6)
        Specifies the *tolerance* of the model.  The model will not apply any
        penalty to predictions whose residual is smaller than this parameter.

    C: float (default 1.4)
        Penalty (regularization) parameter of the error term.
        This parameters controls the degree to which margin should be maximized
        at the expense of accuracy.  For large values of `C` , a smaller margin
        will be accepted if the decision function is better at classifying all
        training points correctly. A lower `C` will encourage a larger margin and
        therefore a simpler decision function.

    gamma: float or string (default 0.001)
        Kernel coefficient for the radial basis function (rbf) kernel.
        This parameter controls the measure of similarity between vectors in the
        training set.

        If a string is passed, then it should be one of 'auto', which uses
        1 / n_features, or 'scale', which uses 1 / (n_features * variance)
    '''
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._estimator = scikit_SVR()

    @property
    def estimator(self):
        return self._estimator

    @property
    def default_hyperparams(self):
        return {
            'C': 1200,
            'epsilon': 2,
            'kernel': 'rbf',
            'gamma': 9e-05
        }
