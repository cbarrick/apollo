from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from apollo.models.scikit_estimator import ScikitModel


class DecisionTree(ScikitModel):
    ''' Model Tree regressor using mean-regression at the leaves

    A Model Tree is a decision tree that predicts a real-valued target instead
    of a categorical class.
    The model tree algorithm used by scikit-learn is identical to a normal
    decision tree used for classification, except the mean of the examples at
    the leaves of the tree is used as the predicted value.

    This model accepts the following hyperparameters, which can be set by
    passing kwargs with matching names to the model's constructor:

    criterion: string (default 'mae')
        Function measuring the quality of a split.  Should be one of 'mse',
        'friedman_mse', or 'mae'.

        - 'mae', mean absolute error, minimizes the L1 loss
        - 'mse', mean squared error, minimizes the L2 loss
        - 'friedman_mse' is similar to 'mse' but uses an improved version of the
        scoring function published by Friedman et al.

    splitter: string (default 'best')
        The splitting strategy for each node.  Should be 'best' or 'random'.

        The 'best' strategy will choose the split that results in the largest
        improvement in the splitting criterion.
        The 'random' strategy will choose the best random split.

    max\_depth: int (default 20)
        Maximum depth of the tree.

        Deeper trees are more prone to overfitting.
        Shallower trees to underfitting.

    min\_impurity\_decrease: float (default 0.25)
        A node will be split if the split causes the impurity of the node to
        decrease by an amount greater than this parameter.

        A minimum impurity decrease is often imposed to avoid overfitting in
        trees that are not pruned post-training.
    '''
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._estimator = DecisionTreeRegressor()

    @property
    def estimator(self):
        return self._estimator

    @property
    def default_hyperparams(self):
        return {
            'criterion': 'mae',
            'splitter': 'random',
            'max_depth': 10,
            'min_impurity_decrease': 0.25
        }


class RandomForest(ScikitModel):
    ''' Random Forest regressor

    A Random Forest is a bagging ensemble of model trees.
    Bagging models combine multiple simple learners (such as decision trees)
    into an emsemble, which improves accuracy if biases of the simple learners
    are not strongly correlated.
    Random forests encourage diversity in the underlying model trees by building
    each tree using a random subset of the features in the input.

    This model accepts the following hyperparameters, which can be set by
    passing kwargs with matching names to the model's constructor:

    n\_estimators: int (default 100)
        The number of trees to include in the ensemble.

    max\_depth: int (default 20)
        Maximum depth of the tree.

        Deeper trees are more prone to overfitting.
        Shallower trees to underfitting.

    min\_impurity\_decrease: float (default 0.25)
        A node will be split if the split causes the impurity of the node to
        decrease by an amount greater than this parameter.

        A minimum impurity decrease is often imposed to avoid overfitting in
        trees that are not pruned post-training.
    '''
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._estimator = RandomForestRegressor()

    @property
    def estimator(self):
        return self._estimator

    @property
    def default_hyperparams(self):
        return {
            'criterion': 'mae',
            'n_estimators': 125,
            'max_depth': 20,
            'min_impurity_decrease': 0.4
        }


class GradientBoostedTrees(ScikitModel):
    ''' Extreme Gradient-Boosted Tree ensembles

    Gradient boosting is a technique for training an ensemble of models.
    The models in the `n`th iteration are trained such that they perform
    better against the examples where the `n-1` iteration performed poorly.
    Extreme Gradient Boosting is a implementation of gradient boosting
    which, according to its author, uses "a more regularized model formalization
    to control over-fitting, which gives it better performance.”

    This model accepts the following hyperparameters, which can be set by
    passing kwargs with matching names to the model's constructor:

    learning\_rate: float (default 0.05)
        Weighting factor for the corrections made in each iteration.

        A lower learning rate can help prevent overfitting, but a higher
        learning rate will cause the models to converge more quickly.

    n\_estimators: int (default 100)
        The number of trees to include in the ensemble.

    max\_depth: int (default 20)
        Maximum depth of the tree.

        Deeper trees are more prone to overfitting.
        Shallower trees to underfitting.
    '''
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._estimator = XGBRegressor()

    @property
    def estimator(self):
        return self._estimator

    @property
    def default_hyperparams(self):
        return {
            'learning_rate': 0.06,
            'n_estimators': 125,
            'max_depth': 5,
        }
