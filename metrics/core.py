class Sum:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.val = 0

    def accumulate(self, batch):
        try:
            self.val += batch.sum(**self.kwargs)
        except (TypeError, AttributeError):
            self.val += batch

    def reduce(self):
        val = self.val
        self.val = 0
        return val


class Mean:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.n = 0
        self.val = 0

    def accumulate(self, batch):
        if hasattr(batch, 'mean'):
            n = len(batch)
            val = batch.mean(**self.kwargs)
        elif hasattr(batch, 'double'):
            n = len(batch)
            val = batch.double().mean(**self.kwargs)
        else:
            n = 1
            val = batch

        if self.n == 0:
            self.n = n
            self.val = val

        else:
            delta = val - self.val
            self.n += n
            self.val += delta * n / self.n

    def reduce(self):
        val = self.val
        self.n = 0
        self.val = 0
        return val
