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
    ds = nam.select_range('2016-12-01', '2017-10-01')
    ds.ga_power.setup(feature='DSWRF_SFC', module=7, column=5)
    input_size, target_size = ds.ga_power.sizes()
    linear = torch.nn.Linear(input_size, target_size).cuda()
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(linear.parameters())
    for i, (x, y) in enumerate(ds.ga_power.load(batch_size=32, pin_memory=True)):
        optimizer.zero_grad()
        x = Variable(x.float().cuda())
        y = Variable(y.float().cuda())
        h = linear.forward(x)
        l = loss(h, y)
        if i % 10 == 0:
            print(f'{i} : {l.data[0]}')
            state = linear.state_dict()
            torch.save(state, f'linear_{i}.pt')
        l.backward()
        optimizer.step()
