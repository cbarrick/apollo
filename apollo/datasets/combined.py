from collections import defaultdict

import numpy as np

from apollo.datasets import nam, ga_power


def open_range(module=7, start='2017-01-01', stop='today', nam_kwargs={}, ga_power_kwargs={}):
    forecast = nam.open_range(start, stop, **nam_kwargs)
    targets = ga_power.open_aggregate(module, **ga_power_kwargs)
    return Joined(forecast, targets)


def join(forecast, target, on='reftime'):
    return Joined(forecast, target, on)


class Joined:
    def __init__(self, forecast, targets, on='reftime'):
        # Only the indexes which are common to both.
        # The inner loop should be smallest. Is it?
        indexes = [t for t in forecast[on].data if t in targets.index]

        self.forecast = forecast.loc[{on: indexes}]
        self.targets = targets.loc[indexes]
        self.on = on
        assert len(self.forecast[on]) == len(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.forecast[{self.on: idx}]
        y = self.targets.iloc[idx]

        # Extract arrays for variables with different shapes (e.g. different z-axis).
        # Combine arrays for variables with same shape (e.g. surface and cloud variables).
        layers = defaultdict(lambda: [])
        for name in sorted(x.data_vars):
            layer = x[name]
            layers[layer.shape].append(layer.data)
        layers = {k:np.stack(v) for k,v in layers.items()}

        # Coalesce z-axis into independent features.
        layers = layers.values()
        layers = [v.transpose(0,2,1,3,4) for v in layers]
        shapes = [v.shape for v in layers]
        shape = shapes[0][2:]
        for s in shapes: assert s[2:] == shape
        layers = [v.reshape((-1, *shape)) for v in layers]

        # Combine all features
        x = np.concatenate(layers)
        y = y.values[17]
        return x, y
