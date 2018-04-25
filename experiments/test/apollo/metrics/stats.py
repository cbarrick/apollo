'''Accumulators for general purpose statistics.

This module provides accumulators like `Mean` which are universally useful.
These general accumulators all accept a `transform` function to apply to each
batch before accumulation. This is handy to compute a wide range of statistics.
For example, if you have a loss function called `loss`, you can create an
accumulator for mean loss with `Mean(loss)`.
'''

import abc

import numpy as np

from apollo.metrics import Accumulator


class Sum(Accumulator):
    '''An accumulator for sums of data.

    Inputs to the accumulator are taken in batches. Each batch is first
    reduced, then the reduced values are added to a running total. This leads
    to three cases:

    1. If a given batch is a scaler, it is added to the accumulator directly.
    2. If a batch has a `sum` method, the batch is reduced using that method.
    3. Otherwise the batch is reduced using numpy's `np.sum` function.

    In the second and third cases, additional keyword arguments passed to the
    constructor are forwarded to either the `.sum` method call or `np.sum`
    function call on each batch. This allows e.g. summing along a specific
    axis.
    '''

    def __init__(self, transform=None, **kwargs):
        '''Initialize a Sum accumulator.

        Args:
            transform (callable):
                Called on each batch before accumulation.
            kwargs:
                Forwarded to the `.sum` method or `np.sum` function
                for each batch.
        '''
        self.transform = transform
        self.kwargs = kwargs
        self.val = None

    def accumulate(self, batch):
        '''input a new batch of data to the accumulator.

        Args:
            batch: A collection of data.
        '''
        if self.transform is not None:
            batch = self.transform(batch)

        if np.isscalar(batch):
            val += batch
        elif hasattr(batch, 'sum'):
            val += batch.sum(**self.kwargs)
        else:
            val += np.sum(batch, **self.kwargs)

        if self.val is None:
            self.val = val
        else:
            self.val += val

    def reduce(self):
        '''Returns the sum of observed data and resets the accumulator.
        '''
        val = self.val
        self.val = 0
        return val


class Mean(Accumulator):
    '''An accumulator for means of data.

    Inputs to the accumulator are taken in batches. Each batch is first
    reduced, then the reduced values are input to a running average. This leads
    to three cases:

    1. If a given batch is a scaler, it is input to the accumulator directly.
    2. If a batch has a `mean` method, the batch is reduced using that method.
    3. Otherwise the batch is reduced using Numpy's `np.mean` function.

    In the second and third cases, additional keyword arguments passed to the
    constructor are forwarded to either the `.mean` method call or `np.mean`
    function call on each batch. This allows e.g. averaging along a specific
    axis.
    '''

    def __init__(self, transform=None, **kwargs):
        '''Initialize a Mean accumulator.

        Args:
            transform (callable):
                Called on each batch before accumulation.
            kwargs:
                Forwarded to the `.mean` method or `np.mean` function
                for each batch.
        '''
        self.transform = transform
        self.kwargs = kwargs
        self.val = 0
        self.n = 0

    def accumulate(self, batch):
        '''input a new batch of data to the accumulator.

        Args:
            batch: A collection of data.
        '''
        if self.transform is not None:
            batch = self.transform(batch)

        if np.isscalar(batch):
            n = 1
            val = batch
        elif hasattr(batch, 'mean'):
            n = len(batch)
            val = batch.mean(**self.kwargs)
        else:
            n = len(batch)
            val = np.mean(batch, **self.kwargs)

        if self.n == 0:
            self.n = n
            self.val = val
        else:
            delta = val - self.val
            self.n += n
            self.val += delta * n / self.n

    def reduce(self):
        '''Returns the mean of observed data and resets the accumulator.
        '''
        val = self.val
        self.n = 0
        self.val = 0
        return val
