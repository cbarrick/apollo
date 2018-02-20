import logging

import numpy as np
import xarray as xr

import torch

import estimators as E
import metrics as M
import networks as N
import optim as O
from datasets import uga_solar


logger = logging.getLogger(__name__)


class SqueezeTime:
    '''A dataset transformation that squeezes the time axis into the feature axis.
    '''
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        (x, y) = self.dataset[i]
        (c, t, h, w) = x.shape
        x = x.reshape((c*t, h, w))
        return (x, y)


def set_seed(n):
    '''Seed the RNGs of stdlib, numpy, and torch.'''
    import random
    import numpy as np
    import torch
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n)


def select(ds, *features):
    data_vars = {f:ds[f] for f in features}
    coords = ds.coords
    return xr.Dataset(data_vars)


def main(name=None, *, epochs=600, learning_rate=0.001, patience=None, batch_size=32,
        start='2017-01-01T00:00', stop='2018-01-01T00:00', target_module=7,
        seed=1337, dry_run=False, log_level='INFO'):

    logging.basicConfig(
        level=log_level,
        style='{',
        format='[{levelname:.4}][{asctime}][{name}:{lineno}] {msg}',
    )

    set_seed(seed)

    torch.set_default_tensor_type('torch.cuda.DoubleTensor')

    # TODO: Accept features and region as args
    features = ('DSWRF_SFC', 'DLWRF_SFC', 'TCC_EATM', 'TMP_SFC', 'VGRD_TOA', 'UGRD_TOA')
    region = {'y':slice(59,124),'x':slice(49,114)}

    forecast = uga_solar.nam.open_range(start, stop)
    forecast = select(forecast, *features)
    forecast = forecast.isel(**region)

    targets = uga_solar.ga_power.open_aggregate(target_module)

    train_set = uga_solar.join(forecast, targets, on='reftime')
    train_set = SqueezeTime(train_set)
    logger.info(f'train set size: {len(train_set)}')

    net = N.Vgg11(shape=(222, 64, 64), ndim=1)  # TODO: Don't hardcode shape
    opt = O.Adam(net.parameters(), lr=learning_rate)
    loss = N.SmoothL1Loss()
    model = E.Classifier(net, opt, loss, name=name, dry_run=dry_run)
    model.fit(train_set, num_workers=0, epochs=epochs, patience=patience, batch_size=batch_size)

    return model
