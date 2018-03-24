'''Metrics for classification tasks.

This module provides accumulators for various classification metrics.
'''

from apollo.metrics import Accumulator
from apollo.metrics.stats import Sum, Mean


# Use epsilon only to prevent ZeroDivisionError.
# Rounding error may exceed epsilon.
import sys
EPSILON = sys.float_info.epsilon


class Accuracy(Accumulator):
    '''An accumulator for accuracy scores.

    Labels and predictions are compared using the `==` operator and then
    forwarded to an underlying `Mean` accumulator.
    '''

    def __init__(self, **kwargs):
        '''Initialize an Accuracy accumulator.

        Args:
            kwargs: Forwarded to the underlying `Mean` accumulator.
        '''
        self.val = Mean(**kwargs)

    def accumulate(self, y, h):
        '''Accumulate batches of labels and predictions.

        Args:
            y: A batch of labels.
            h: A batch of predictions.
        '''
        val = (y == h)
        self.val.accumulate(val)

    def reduce(self):
        '''Returns the mean accuracy of observed data and resets the
        accumulator to its initial state.
        '''
        return self.val.reduce()


class TruePositives(Accumulator):
    '''An accumulator for true positive counts.

    Labels and predictions are compared to the target using the `==` operator
    then combined with the `&` operator. The result is forwarded to a `Sum`
    accumulator.
    '''

    def __init__(self, target=1, **kwargs):
        '''Initialize a TruePositives accumulator.

        Args:
            target: The value of the positive class.
            kwargs: Forwarded to the underlying `Sum` accumulator.
        '''
        self.target = target
        self.val = Sum(**kwargs)

    def accumulate(self, y, h):
        '''Accumulate batches of labels and predictions.

        Args:
            y: A batch of labels.
            h: A batch of predictions.
        '''
        val = (h == self.target) & (y == self.target)
        self.val.accumulate(val)

    def reduce(self):
        '''Returns the true positive count of observed data and resets the
        accumulator to its initial state.
        '''
        return self.val.reduce()


class FalsePositives(Accumulator):
    '''An accumulator for false positive counts.

    Labels and predictions are compared to the target using the `==` operator
    then combined with the `&` operator. The result is forwarded to a `Sum`
    accumulator.
    '''

    def __init__(self, target=1, **kwargs):
        '''Initialize a FalsePositives accumulator.

        Args:
            target: The value of the positive class.
            kwargs: Forwarded to the underlying `Sum` accumulator.
        '''
        self.target = target
        self.val = Sum(**kwargs)

    def accumulate(self, y, h):
        '''Accumulate batches of labels and predictions.

        Args:
            y: A batch of labels.
            h: A batch of predictions.
        '''
        val = (h == self.target) & (y != self.target)
        self.val.accumulate(val)

    def reduce(self):
        '''Returns the false positive count of observed data and resets the
        accumulator to its initial state.
        '''
        return self.val.reduce()


class TrueNegatives(Accumulator):
    '''An accumulator for true negatives counts.

    Labels and predictions are compared to the target using the `==` operator
    then combined with the `&` operator. The result is forwarded to a `Sum`
    accumulator.
    '''

    def __init__(self, target=1, **kwargs):
        '''Initialize a TrueNegatives accumulator.

        Args:
            target: The value of the positive class.
            kwargs: Forwarded to the underlying `Sum` accumulator.
        '''
        self.target = target
        self.val = Sum(**kwargs)

    def accumulate(self, y, h):
        '''Accumulate batches of labels and predictions.

        Args:
            y: A batch of labels.
            h: A batch of predictions.
        '''
        val = (h != self.target) & (y != self.target)
        self.val.accumulate(val)

    def reduce(self):
        '''Returns the true negative count of observed data and resets the
        accumulator to its initial state.
        '''
        return self.val.reduce()


class FalseNegatives(Accumulator):
    '''An accumulator for false negatives counts.

    Labels and predictions are compared to the target using the `==` operator
    then combined with the `&` operator. The result is forwarded to a `Sum`
    accumulator.
    '''

    def __init__(self, target=1, **kwargs):
        '''Initialize a FalseNegatives accumulator.

        Args:
            target: The value of the positive class.
            kwargs: Forwarded to the underlying `Sum` accumulator.
        '''
        self.target = target
        self.val = Sum(**kwargs)

    def accumulate(self, y, h):
        '''Accumulate batches of labels and predictions.

        Args:
            y: A batch of labels.
            h: A batch of predictions.
        '''
        val = (h != self.target) & (y == self.target)
        self.val.accumulate(val)

    def reduce(self):
        '''Returns the false negative count of observed data and resets the
        accumulator to its initial state.
        '''
        return self.val.reduce()


class Precision(Accumulator):
    '''An accumulator for precision scores.

    Labels and predictions are compared to the target using the `==` operator
    then combined with the `&` operator.
    '''

    def __init__(self, target=1, **kwargs):
        '''Initialize a Precision accumulator.

        Args:
            target: The value of the positive class.
            kwargs: Forwarded to the underlying `Sum` accumulators.
        '''
        self.tp = TruePositives(target, **kwargs)
        self.fp = FalsePositives(target, **kwargs)

    def accumulate(self, y, h):
        '''Accumulate batches of labels and predictions.

        Args:
            y: A batch of labels.
            h: A batch of predictions.
        '''
        self.tp.accumulate(y, h)
        self.fp.accumulate(y, h)

    def reduce(self):
        '''Returns the precision of observed data and resets the
        accumulator to its initial state.
        '''
        tp = self.tp.reduce()
        fp = self.fp.reduce()
        return tp / (tp + fp + EPSILON)


class Recall(Accumulator):
    '''An accumulator for recall scores.

    Labels and predictions are compared to the target using the `==` operator
    then combined with the `&` operator.
    '''

    def __init__(self, target=1, **kwargs):
        '''Initialize a Recall accumulator.

        Args:
            target: The value of the positive class.
            kwargs: Forwarded to the underlying `Sum` accumulators.
        '''
        self.tp = TruePositives(target, **kwargs)
        self.fn = FalseNegatives(target, **kwargs)

    def accumulate(self, y, h):
        '''Accumulate batches of labels and predictions.

        Args:
            y: A batch of labels.
            h: A batch of predictions.
        '''
        self.tp.accumulate(y, h)
        self.fn.accumulate(y, h)

    def reduce(self):
        '''Returns the recall of observed data and resets the
        accumulator to its initial state.
        '''
        tp = self.tp.reduce()
        fn = self.fn.reduce()
        return tp / (tp + fn + EPSILON)


class FScore(Accumulator):
    '''An accumulator for F-scores.

    Labels and predictions are compared to the target using the `==` operator
    then combined with the `&` operator.
    '''

    def __init__(self, beta=1, target=1, **kwargs):
        '''Initialize an FScore accumulator.

        Args:
            beta: The f-score parameter.
            target: The value of the positive class.
            kwargs: Forwarded to the underlying `Sum` accumulators.
        '''
        self.beta = beta
        self.tp = TruePositives(target, **kwargs)
        self.fp = FalsePositives(target, **kwargs)
        self.fn = FalseNegatives(target, **kwargs)

    def accumulate(self, y, h):
        '''Accumulate batches of labels and predictions.

        Args:
            y: A batch of labels.
            h: A batch of predictions.
        '''
        self.tp.accumulate(y, h)
        self.fp.accumulate(y, h)
        self.fn.accumulate(y, h)

    def reduce(self):
        '''Returns the f-score of observed data and resets the
        accumulator to its initial state.
        '''
        tp = self.tp.reduce()
        fp = self.fp.reduce()
        fn = self.fn.reduce()
        beta2 = self.beta ** 2
        tp2 = (1 + beta2) * tp
        fn2 = beta2 * fn
        return tp2 / (tp2 + fn2 + fp + EPSILON)
