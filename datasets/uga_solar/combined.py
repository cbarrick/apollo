from collections import defaultdict

import numpy as np

from datasets.uga_solar import nam, ga_power


def open_range(module=7, start='2017-01-01', stop='today', nam_kwargs={}, ga_power_kwargs={}):
    nam_data = nam.open_range(start, stop, **nam_kwargs)
    ga_power_data = ga_power.open_aggregate(module, **ga_power_kwargs)
    return Joined(nam_data, ga_power_data)


class Joined:
    def __init__(self, nam_data, ga_power_data, key='reftime'):
        # Only the indexes which are common to both.
        # The inner loop should be smallest. Is it?
        indexes = [t for t in nam_data[key].data if t in ga_power_data.index]

        self.nam = nam_data.loc[{key: indexes}]
        self.ga_power = ga_power_data.loc[indexes]
        self.key = key
        assert len(self.nam[key]) == len(self.ga_power)

    def __getitem__(self, idx):
        ds = self.nam[{self.key: idx}]
        df = self.ga_power.iloc[idx]

        layers = defaultdict(lambda: [])
        for name in sorted(ds.data_vars):
            layer = ds[name]
            layers[layer.shape].append(layer.data)

        x = [np.stack(layers[name]) for name in sorted(layers)]
        y = df.iloc[idx]
        return x, y

    def __len__(self):
        return len(self.ga_power)
