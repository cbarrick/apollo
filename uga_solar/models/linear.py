import logging

import numpy as np
import xarray as xr

import torch
from torch.autograd import Variable

from uga_solar.data import nam


# Module level logger
logger = logging.getLogger(__name__)


class DataHelper:
    def __init__(self, start='20160901T0000', stop='20170831T1800', target_column=5):
        self.x = nam.load_range(start, stop)
        self.y = nam.load_targets(7)
        self.size = len(self.x['reftime'])
        self._targets = torch.zeros(37)
        self._targets = Variable(self._targets, requires_grad=False)
        self.target_column = target_column

    def summary(self):
        logger.info('computing summary statistics')
        mean = self.x.mean().compute()
        std = self.x.std().compute()
        return {
            'mean': {name: mean[name].values[None] for name in self.x.data_vars},
            'std': {name: std[name].values[None] for name in self.x.data_vars}
        }

    def quick_summary(self, n=100):
        logger.info(f'computing summary statistics from a sample of {n}')
        l = len(self.x.reftime)
        i = np.random.randint(n, size=l)
        x = self.x.isel(reftime=i)
        mean = x.mean()
        std = x.std()
        return {
            'mean': {name: mean[name].values[None] for name in self.x.data_vars},
            'std': {name: std[name].values[None] for name in self.x.data_vars}
        }

    def isel(self, i):
        x = self.x.isel(reftime=i)
        reftime = self.x.reftime[i].values

        try:
            for j in range(37):
                ftime = reftime + x.forecast[j].values
                self._targets.data[j] = self.y.loc[ftime][self.target_column]
        except KeyError:
            logger.warning(f'no targets for {reftime}')
            return self.random_batch()

        return x, self._targets

    def iter(self):
        n = len(self.x.reftime)
        for i in range(n):
            yield self.isel(i)

    def random_batch(self):
        i = np.random.randint(self.size)
        return self.isel(i)


class SingleLinear(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.register_buffer('mean', torch.zeros(1))
        self.register_buffer('std', torch.ones(1))
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        x = x - Variable(self.mean)
        x = x / Variable(self.std)
        return self.linear(x)

    def set_scale(self, mean, std):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)


class CombinedLinear:
        def __init__(self, template):
            self.is_cuda = False
            self._module_names = []
            self._modules = torch.nn.ModuleList()
            for name in template.data_vars:
                shape = template[name].shape
                n_features = int(np.prod(shape[1:]))
                if n_features > 0:
                    mod = SingleLinear(int(n_features))
                    self._modules.append(mod)
                    self._module_names.append(name)

        @property
        def modules(self):
            return zip(self._module_names, self._modules)

        @property
        def parameters(self):
            return self._modules.parameters()

        def cuda(self):
            self.is_cuda = True
            self._modules = self._modules.cuda()
            return self

        def save(self, path):
            state = self._modules.state_dict()
            return torch.save(state, str(path))

        def load(self, path):
            state = torch.load(str(path))
            return self._modules.load_state_dict(state)

        def set_scale(self, helper):
            summary = helper.quick_summary()
            for name, mod in self.modules:
                mod.set_scale(summary['mean'][name], summary['std'][name])

        def predict(self, ds):
            n_samples = len(ds['forecast'])
            parts = []
            for name, mod in self.modules:
                var = ds.data_vars[name]
                arr = var.values.astype('float32')
                arr = arr.reshape(n_samples, -1)
                arr = torch.Tensor(arr)
                arr = Variable(arr, requires_grad=True)
                if self.is_cuda:
                    arr = arr.cuda()
                h = mod(arr)
                parts.append(h)
            return sum(parts)

        def mse(self, helper):
            score = 0
            size = 0
            for x, y in helper.iter():
                n = len(y)
                if self.is_cuda:
                    y = y.cuda()
                h = self.predict(x)
                err = (h - y) ** 2 / n
                if size == 0:
                    score = err
                    size = n
                else:
                    score += (err - score) * n / size
                    n += size
            return score

        def partial_fit(self, helper, optimizer, loss):
            optimizer.zero_grad()
            x, y = helper.random_batch()
            if self.is_cuda:
                y = y.cuda()
            h = self.predict(x)
            l = loss(h, y)
            l.backward()
            optimizer.step()

        def fit(self, helper, n_iters, **kwargs):
            loss = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(self.parameters, **kwargs)
            for i in range(n_iters):
                self.partial_fit(helper, optimizer, loss)


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')
    train = DataHelper('20160901T0000', '20170831T1800')
    test  = DataHelper('20170901T0000', '20170929T1800')
    template, _ = train.random_batch()
    model = CombinedLinear(template).cuda()

    # # # Setting the scale takes 2+ hours.
    # # # Save the result so that we don't do it every time.
    # # model.set_scale(train)
    # # model.save('untrained.pt')
    # model.load('untrained.pt')
    #
    # logger.info('training')
    # for i in range(1000000):
    #     model.fit(train, 100, lr=0.001)
    #     model.save(f'linear_{i}.pt')
    #     mse = model.mse(test)
    #     logger.info(f'test mse: {mse}')
