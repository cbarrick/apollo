import logging

import numpy as np
import xarray as xr

import torch

from datasets.uga_solar import nam
from datasets.uga_solar import ga_power


# Module level logger
logger = logging.getLogger(__name__)


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(len(x), -1)


class SimpleNet(torch.nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        n_features = int(np.prod(in_shape))

        self._module = torch.nn.Sequential(
            Flatten(),
            torch.nn.Linear(n_features, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

        torch.nn.init.kaiming_uniform(self._module[1].weight)
        torch.nn.init.kaiming_uniform(self._module[3].weight)

    def forward(self, x):
        x = x.permute(0, 4, 2, 3, 1)
        x = x.squeeze()
        y = self._module(x)
        return y


class ForecastToTruth:
    def __init__(self,
            criteria=torch.nn.MSELoss(),
            dtype=None,
        ):

        self.criteria = criteria
        if dtype is None:
            if torch.cuda.is_available():
                self.dtype = torch.cuda.FloatTensor
            else:
                self.dtype = torch.FloatTensor
        else:
            self.dtype = dtype

        self.module = None
        self.optimizer = None

    def predict(self, x):
        # TODO: handle multiple x, garuntee the ordering of features
        x = torch.autograd.Variable(x)
        return self.module(x)

    def score(self, x, y):
        x = x.type(self.dtype)
        y = y.type(self.dtype)
        h = self.predict(x)
        y = torch.autograd.Variable(y)
        return self.criteria(h, y)

    def fit_epoch(self, data_loader):
        mean_loss = 0
        n = 0

        for x, y in data_loader:
            # initialize the module
            if self.module == None:
                self.module = SimpleNet(x.shape[1:])
                self.optimizer = torch.optim.Adam(self.module.parameters())

            # train step
            print('.', end='', flush=True)
            self.optimizer.zero_grad()
            self.module.train(True)
            loss = self.score(x, y)
            loss.backward()
            self.optimizer.step()
            loss = loss.data.numpy()[0]
            delta = loss - mean_loss
            n += len(y)
            mean_loss += delta * len(y) / n

        return mean_loss

    def fit(self, data_loader):
        best_loss = float('inf')
        for epoch in range(100):
            print(f'epoch {epoch}', end=' ', flush=True)
            loss = self.fit_epoch(data_loader)
            if loss < best_loss:
                best_loss = loss
                state = self.model.state_dict()
                torch.save(state, f'forecast_to_truth_{epoch}.pt')
            print(f'DONE (mse = {loss:.3e})')
        return best_loss


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG', format='[{asctime}] {levelname:>7}: {message}', style='{')

    features = ('DSWRF_SFC', 'DLWRF_SFC')
    start = '2016-12-01T00:00'
    stop = '2017-10-01T00:00'
    target_col = 17
    center = (91,81)
    window = 3

    forecasts = nam.open_range(start, stop)
    irradiance = ga_power.open_aggregate(7)[target_col].dropna()
    y_lo, y_hi = center[0] - window//2, center[0] + window//2 + 1
    x_lo, x_hi = center[1] - window//2, center[1] + window//2 + 1

    train_set = (
        forecasts.torch
            .key('reftime')
            .select(*features)
            .where(forecast=0)
            .iwhere(y=slice(y_lo, y_hi))
            .iwhere(x=slice(x_lo, x_hi))
            .join(irradiance)
    )

    model = ForecastToTruth()
    train_loss = model.fit(train_set)
    print(f'Best MSE: {train_loss}')
