from collections import defaultdict

import numpy as np

from datasets.uga_solar import nam, ga_power


def open_range(module=7, start='2017-01-01', stop='today', nam_kwargs={}, ga_power_kwargs={}):
    nam_data = nam.open_range(start, stop, **nam_kwargs)
    ga_power_data = ga_power.open_aggregate(module, **ga_power_kwargs)
    return Joined(nam_data, ga_power_data)


def join(forecast, target, on='reftime'):
    return Joined(forecast, target, on)


class Joined:
    def __init__(self, nam_data, ga_power_data, on='reftime'):
        # Only the indexes which are common to both.
        # The inner loop should be smallest. Is it?
        indexes = [t for t in nam_data[on].data if t in ga_power_data.index]

        self.nam = nam_data.loc[{on: indexes}]
        self.ga_power = ga_power_data.loc[indexes]
        self.on = on
        assert len(self.nam[on]) == len(self.ga_power)

    def __getitem__(self, idx):
        ds = self.nam[{self.on: idx}]
        df = self.ga_power.iloc[idx]

        # Extract arrays for variables with different shapes (e.g. different z-axis).
        # Combine arrays for variables with same shape (e.g. surface and cloud variables).
        layers = defaultdict(lambda: [])
        for name in sorted(ds.data_vars):
            layer = ds[name]
            layers[layer.shape].append(layer.data)
        layers = {k:np.stack(v) for k,v in layers.items()}

        # Coalesce z-axis into independent features.
        layers = layers.values()
        layers = [v.transpose(0,2,1,3,4) for v in layers]
        shapes = [v.shape for v in layers]
        shape = shapes[0][2:]
        for s in shapes: assert s[2:] == shape
        layers = [v.reshape((-1, *shape)) for v in layers]

        # Combine all features and get corresponding labels.
        x = np.concatenate(layers)
        y = df.iloc[idx]
        return x, y

    def __len__(self):
        return len(self.ga_power)
