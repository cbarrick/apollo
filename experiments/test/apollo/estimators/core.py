import logging
import sys
from pathlib import Path

import numpy as np
import torch

from torch import nn
from torch import autograd as A
from torch import utils.data as D
from torch import nn.functional as F

from apollo.metrics.stats import Mean


logger = logging.getLogger(__name__)


class _DataParallel(nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__()
        self.module = module
        self.parallel = nn.parallel.DataParallel(module, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.parallel(*args, **kwargs)

    def reset(self):
        self.module.reset()
        return self


class Estimator:
    '''Wraps a torch network, optimizer, and loss to an sklearn-like estimator.
    '''

    def __init__(self, net, opt, loss, name='model', cuda=None, dry_run=False):
        '''Create basic estimator.

        Args:
            net: The network to train.
            opt: The optimizer to step during training.
            loss: The loss function to minimize.
            name: A name for the estimator.
            cuda: List of cuda device ids, defaults to all devices.
            dry_run: Cut loops short, useful for debugging.
        '''
        self.cuda = cuda or list(range(torch.cuda.device_count()))
        self._net = self.setup_network(net)
        self._opt = opt
        self._loss = loss
        self.name = name
        self.dry_run = dry_run
        self.reset()

        self.path = Path(f'./checkpoints/{self.name}.torch')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()

        logger.debug(f'created estimator {self.name}')
        logger.debug(f' cuda: {self.cuda}')
        logger.debug(f' network: {self._net.__class__}')
        logger.debug(f' optimizer: {self._opt.__class__}')
        logger.debug(f' loss: {self._loss.__class__}')
        logger.debug(f' dry_run: {self.dry_run}')

    @property
    def net(self):
        '''The network used by the estimator.
        '''
        return self._net

    @property
    def opt(self):
        '''The optimizer used by the estimator.
        '''
        return self._opt

    @property
    def loss(self):
        '''The loss function used by the estimator.
        '''
        return self._loss

    def setup_network(self, net):
        if self.cuda:
            net = net.cuda()
            return _DataParallel(net, device_ids=self.cuda)
        else:
            return net

    def reset(self):
        '''Reset the estimator to it's initial state.

        TODO: This does NOT reset the optimizer.
        I am still deciding the best way to do that.

        Returns:
            Returns `self` to allow method chaining.
        '''
        self.net.reset()
        return self

    def save(self, path=None):
        '''Saves the model parameters to disk.

        The default path is based on the name of the estimator,
        and can be overridden by manipulating `esimator.path`.

        Args:
            path: The path to write into.

        Returns:
            Returns `self` to allow method chaining.
        '''
        if path is None: path = self.path
        logger.debug(f'saving {self.name} to {path}')
        state = self.net.state_dict()
        torch.save(state, str(path))
        return self

    def load(self, path=None):
        '''Loads the model parameters from disk.

        The default path is based on the name of the estimator,
        and can be overridden by manipulating `esimator.path`.

        Args:
            path: The path to write into.

        Returns:
            Returns `self` to allow method chaining.
        '''
        if path is None: path = self.path
        logger.debug(f'restoring {self.name} from {path}')
        state = torch.load(str(path))
        self.net.load_state_dict(state)
        return self

    def variable(self, x, **kwargs):
        '''Cast a tensor to a `Variable` on the same device as the network.

        If the input is already a `Variable`, it is not wrapped,
        but it may be copied to a new device.

        Args:
            x: The tensor to wrap.

        Kwargs:
            Forwarded to the `autograd.Variable` constructor.

        Returns:
            An `autograd.Variable` on the same device as the network.
        '''
        if not isinstance(x, A.Variable):
            x = A.Variable(x, **kwargs)
        if self.cuda:
            x = x.cuda(async=True)
        return x

    def tensor(self, x):
        '''Cast some `x` to a `Tensor` on the same device as the network.

        If `x` is an `autograd.Variable`, then a clone of its data tensor is
        returned. Otherwise `x` is passed to the `torch.Tensor` constructor.

        Args:
            x: The input to cast.

        Returns:
            A `torch.Tensor` on the same device as the network.
        '''
        if isinstance(x, A.Variable):
            x = x.data.clone()
        elif hasattr(x, 'cuda'):
            # x is already a Tensor if it has a cuda method (probably).
            # There is no common base class for Tensors as of PyTorch 0.2.
            pass
        else:
            x = torch.Tensor(x)
        if self.cuda:
            x = x.cuda(async=True)
        return x

    def params(self):
        '''Get the list of trainable paramaters.

        Returns:
            A list of all parameters in the optimizer's `param_groups`.
        '''
        return [p for group in self.opt.param_groups for p in group['params']]

    def partial_fit(self, x, y):
        '''Performs one step of the optimization.

        Args:
            x: The input batch.
            y: The class labels.

        Returns:
            Returns the average loss for this batch.
        '''
        self.net.train()
        self.opt.zero_grad()
        x = self.variable(x)
        y = self.variable(y)
        h = self.net(x)
        j = self.loss(h, y)
        j_mean = j.data.mean()

        if np.isnan(j_mean):
            logger.warn('Gradient is NaN')
            self.opt.zero_grad()
        else:
            j.backward()
            self.opt.step()

        return j_mean

    def fit(self, train, validation=None, epochs=100, patience=None, reports={}, **kwargs):
        '''Fit the model to a dataset.

        If a validation set is given, all reports and early stopping use it.

        Args:
            train: A dataset to fit.
            validation: A dataset to use as the validation set.
            epochs: The maximum number of epochs to spend training.
            patience: Stop if the loss does not improve after this many epochs.
            reports: A dict of metrics to report at each epoch.

        Kwargs:
            Forwarded to torch's `DataLoader` class, except:
            shuffle: Defaults to True.
            pin_memory: Defaults to True if the estimator is using cuda.
            num_workers: Defaults to 4.

        Returns:
            Returns the validation loss.
            Returns train loss if no validation set is given.
        '''
        kwargs.setdefault('shuffle', True)
        kwargs.setdefault('pin_memory', self.cuda)
        kwargs.setdefault('num_workers', 4)
        train = D.DataLoader(train, **kwargs)

        best_loss = float('inf')
        p = patience or -1
        for epoch in range(epochs):

            # Training
            n = len(train)
            train_loss = Mean()
            print(f'epoch {epoch+1} [0%]', end='\r', flush=True, file=sys.stderr)
            for i, (x, y) in enumerate(train):
                j = self.partial_fit(x, y)
                if not np.isnan(j): train_loss.accumulate(j)
                progress = (i+1) / n
                print(f'epoch {epoch+1} [{progress:.2%}]', end='\r', flush=True, file=sys.stderr)
                if self.dry_run: break
            train_loss = train_loss.reduce()
            print('\001b[2K', end='\r', flush=True, file=sys.stderr)  # magic to clear the line
            print(f'epoch {epoch+1}', end=' ', flush=True)
            print(f'[train loss: {train_loss:8.6f}]', end=' ', flush=True)

            # Validation
            if validation:
                val_loss = self.test(validation, **kwargs)
                print(f'[validation loss: {val_loss:8.6f}]', end=' ', flush=True)

            # Reporting
            for name, criteria in reports.items():
                data = validation if validation else train
                score = self.test(data, criteria, **kwargs)
                print(f'[{name}: {score:8.6f}]', end=' ', flush=True)

            # Early stopping
            loss = val_loss if validation else train_loss
            if loss < best_loss:
                best_loss = loss
                p = patience or -1
                self.save()
                print('✓')
            else:
                p -= 1
                print()
            if p == 0:
                break

        # Revert to best model if using early stopping.
        if patience:
            self.load()

        return loss

    def predict(self, x):
        '''Apply the network to some input batch.

        Args:
            x: The input batch.

        Returns:
            Returns the output of the network.
        '''
        self.net.eval()
        x = self.variable(x, volatile=True)
        h = self.net(x)
        return h.data

    def score(self, x, y, criteria=None):
        '''Score the model on a batch of inputs and labels.

        Args:
            x: The input batch.
            y: The targets.
            criteria: The metric to measure.

        Returns:
            If criteria is None, returns the loss.
            If criteria is a function, returns `criteria(y, h)`.
            If criteria is an accumulator, returns `criteria.accumulate(y, h)`.
        '''
        self.net.eval()

        if criteria is None:
            # Default - the loss has a different signature than a metric
            x = self.variable(x, volatile=True)
            y = self.variable(y, volatile=True)
            h = self.net(x)
            j = self.loss(h, y)
            return j.data

        if not callable(criteria):
            # Accumulator - metrics that can't be simply averaged, like f-score
            h = self.predict(x)
            y = self.tensor(y)
            return criteria.accumulate(y, h)

        try:
            # Function, fast path - try to use the GPU
            h = self.predict(x)
            y = self.tensor(y)
            return criteria(y, h)

        except (ValueError, TypeError):
            # Function, slow path - most sklearn metrics cannot handle Tensors
            logger.debug('computing a metric on the CPU')
            h = h.cpu().numpy()
            y = y.cpu().numpy()
            return criteria(y, h)

    def test(self, data, criteria=None, **kwargs):
        '''Score the model on a dataset.

        Args:
            data: A dataset to score against.
            criteria: The metric to measure; defaults to the loss.

        Kwargs:
            Forwarded to torch's `DataLoader` class, except:
            pin_memory: Defaults to True if the estimator is using cuda.
            num_workers: Defaults to 4.

        Returns:
            If criteria is None, returns the mean loss over all batches.
            If criteria is a function, returns the mean `criteria(y, h)` over all batches.
            If criteria is an accumulator, returns the accumulated metric over all batches.
        '''
        kwargs.setdefault('pin_memory', self.cuda)
        kwargs.setdefault('num_workers', 4)
        data = D.DataLoader(data, **kwargs)

        if criteria is None or callable(criteria):
            # average criteria across batches
            mean = Mean()
            for x, y in data:
                j = self.score(x, y, criteria)
                mean.accumulate(j)
                if self.dry_run: break
            return mean.reduce()

        else:
            # criteria is an accumulator
            # e.g. a metric that can't be simply averaged, like f-score
            for x, y in data:
                self.score(x, y, criteria)
                if self.dry_run: break
            return criteria.reduce()


class Classifier(Estimator):
    '''Wraps a torch network, optimizer, and loss to an sklearn-like classifier.
    '''

    def predict(self, x):
        '''Classify some input batch.

        Args:
            x: The input batch.

        Returns:
            The argmax of the network output.
        '''
        self.net.eval()
        x = self.variable(x, volatile=True)
        h = self.net(x)
        _, h = h.max(1)
        return h.data

    def predict_proba(self, x):
        '''Compute the likelihoods for some input batch.

        Args:
            x: The input batch.

        Returns:
            The softmax of the network output.
        '''
        self.net.eval()
        x = self.variable(x, volatile=True)
        h = self.net(x)
        h = F.softmax(h)
        return h.data
