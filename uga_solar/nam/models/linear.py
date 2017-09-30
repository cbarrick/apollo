import nam.data

class Batcher:
    def __init__(self, ds, batch_size=32):
        self.ds = ds

        n_weights = 0
        for v in self.ds.data_vars.values:
            n_weights += np.product(v.shape[2:])
        self._batch = np.zeros((batch_size, n_weights))

    def batch(self):
        batch_size = self._batch.shape[0]
        n_weights = self._batch.shape[1]
        for i in range(batch_size):
            j = 0
            while j < n_weights:
                pass
