import logging

import numpy as np
import xarray as xr

import torch

from uga_solar.data import nam
from uga_solar.data import ga_power


# Module level logger
logger = logging.getLogger(__name__)


class FullLinear(torch.nn.Module):
    '''A linear layer that opperates on the flattened input.

    This differs from `torch.nn.Linear` in that the latter only operates on
    the rightmost dimension (effectivly making it a 1x1 convolution).
    '''
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.in_shape = tuple(in_shape)
        self.out_shape = tuple(out_shape)
        self.in_size = int(np.prod(in_shape))
        self.out_size = int(np.prod(out_shape))
        self.linear = torch.nn.Linear(self.in_size, self.out_size)

    def forward(self, x):
        n = len(x)
        x = x.view(n, self.in_size)
        y = self.linear(x)
        y = y.view(n, *self.out_shape)
        return y


class MultiLinear(torch.nn.Module):
    def __init__(self, *shapes):
        super().__init__()
        self.in_shapes = shapes[:-1]
        self.out_shape = shapes[-1]
        self.mods = [FullLinear(shape, self.out_shape) for shape in self.in_shapes]
        self.mods = torch.nn.ModuleList(self.mods)

    def forward(self, *inputs):
        n = len(inputs)
        y = sum(mod(x) for mod, x in zip(self.mods, inputs))
        return y


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG', format='[{asctime}] {levelname:>7}: {message}', style='{')

    train = nam.open_range('2016-12-01', '2017-10-01')
    targets = ga_power.open_aggregate(7)[17].dropna()
    target_shape = targets.shape[1:]
    target_size = int(np.prod(target_shape))
    neighborhood = 1
    use_gpu = torch.cuda.is_available()

    sfc_features = ['DSWRF_SFC', 'DLWRF_SFC']
    sfc_shape = (len(sfc_features), train['z_SFC'].size, neighborhood*2, neighborhood*2)

    train_set = (
        train.torch
            .select(*sfc_features)
            .where(forecast=0)
            .iwhere(x=slice(81-neighborhood, 81+neighborhood)) # location of the solar farm
            .iwhere(y=slice(91-neighborhood, 91+neighborhood)) # location of the solar farm
            .join(targets, on='reftime')
    )

    model = MultiLinear(sfc_shape, target_shape)
    if use_gpu:
        model.cuda()

    def score(x, y):
        x = torch.autograd.Variable(x).float()
        y = torch.autograd.Variable(y).float()
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        h = model(x)
        loss = torch.nn.MSELoss()
        return loss(h, y)

    i = 0
    params = model.parameters()
    optimizer = torch.optim.Adam(params)
    for epoch in range(100):
        print(f'epoch {epoch}', end=' ')
        mean_loss = 0
        n = 0
        for x_sfc, y in train_set:
            print('.', end='', flush=True)
            optimizer.zero_grad()
            loss = score(x_sfc, y)
            loss.backward()
            optimizer.step()
            loss = loss.data.numpy()[0]
            delta = loss - mean_loss
            n += len(y)
            mean_loss += delta * len(y) / n
        state = model.state_dict()
        torch.save(state, f'linear_{epoch}.pt')
        print(f'DONE (mse = {mean_loss:.3e})')
