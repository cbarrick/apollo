import sys

import metrics as M


# Use epsilon only to prevent ZeroDivisionError.
# Rounding error may exceed epsilon.
EPSILON = sys.float_info.epsilon


class Accuracy:
    def __init__(self, **kwargs):
        self.val = M.Mean(**kwargs)

    def accumulate(self, y, h):
        val = (y == h)
        val = val.long()
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class TruePositives:
    def __init__(self, target=1, **kwargs):
        self.target = target
        self.val = M.Sum(**kwargs)

    def accumulate(self, y, h):
        val = (h == self.target) & (y == self.target)
        val = val.long()
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class FalsePositives:
    def __init__(self, target=1, **kwargs):
        self.target = target
        self.val = M.Sum(**kwargs)

    def accumulate(self, y, h):
        val = (h == self.target) & (y != self.target)
        val = val.long()
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class TrueNegatives:
    def __init__(self, target=1, **kwargs):
        self.target = target
        self.val = M.Sum(**kwargs)

    def accumulate(self, y, h):
        val = (h != self.target) & (y != self.target)
        val = val.long()
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class FalseNegatives:
    def __init__(self, target=1, **kwargs):
        self.target = target
        self.val = M.Sum(**kwargs)

    def accumulate(self, y, h):
        val = (h != self.target) & (y == self.target)
        val = val.long()
        self.val.accumulate(val)

    def reduce(self):
        return self.val.reduce()


class Precision:
    def __init__(self, target=1, **kwargs):
        self.tp = TruePositives(target, **kwargs)
        self.fp = FalsePositives(target, **kwargs)

    def accumulate(self, y, h):
        self.tp.accumulate(y, h)
        self.fp.accumulate(y, h)

    def reduce(self):
        tp = self.tp.reduce()
        fp = self.fp.reduce()
        return tp / (tp + fp + EPSILON)


class Recall:
    def __init__(self, target=1, **kwargs):
        self.tp = TruePositives(target, **kwargs)
        self.fn = FalseNegatives(target, **kwargs)

    def accumulate(self, y, h):
        self.tp.accumulate(y, h)
        self.fn.accumulate(y, h)

    def reduce(self):
        tp = self.tp.reduce()
        fn = self.fn.reduce()
        return tp / (tp + fn + EPSILON)


class FScore:
    def __init__(self, beta=1, target=1, **kwargs):
        self.beta = beta
        self.tp = TruePositives(target, **kwargs)
        self.fp = FalsePositives(target, **kwargs)
        self.fn = FalseNegatives(target, **kwargs)

    def accumulate(self, y, h):
        self.tp.accumulate(y, h)
        self.fp.accumulate(y, h)
        self.fn.accumulate(y, h)

    def reduce(self):
        tp = self.tp.reduce()
        fp = self.fp.reduce()
        fn = self.fn.reduce()
        beta2 = self.beta ** 2
        tp2 = (1 + beta2) * tp
        fn2 = beta2 * fn
        return tp2 / (tp2 + fn2 + fp + EPSILON)
