import logging

import numpy as np
import xarray as xr

import torch
from torch.autograd import Variable

from uga_solar.data import nam
from uga_solar.data import ga_power


# Module level logger
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG', format='[{asctime}] {levelname:>7}: {message}', style='{')

    train = nam.open_range('2016-12-01', '2017-10-01')
    targets = ga_power.open_aggregate(7)[17]
    target_shape = targets.shape[1:]
    target_size = int(np.prod(target_shape))

    sfc_features = ['DSWRF_SFC', 'DLWRF_SFC']
    sfc_shape = (len(sfc_features), train['z_SFC'].size, train['y'].size, train['x'].size)
    sfc_size = int(np.prod(sfc_shape))
    sfc_linear = torch.nn.Linear(sfc_size, target_size)

    train_set = (
        train.torch
        .select(*sfc_features)
        .where(forecast=0)
        .join(targets, on='reftime')
    )

    def score(x_sfc, y):
        x_sfc = x_sfc.view(-1, sfc_size)
        x_sfc = x_sfc.float()
        x_sfc = Variable(x_sfc)
        y = y.float()
        y = Variable(y)
        h = sfc_linear.forward(x_sfc)
        loss = torch.nn.MSELoss()
        return loss(h, y)

    i = 0
    optimizer = torch.optim.Adam(sfc_linear.parameters())
    for epoch in range(100):
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
        state = sfc_linear.state_dict()
        torch.save(state, f'linear_{epoch}.pt')
        print(f'{epoch} : {mean_loss:.3e}')
