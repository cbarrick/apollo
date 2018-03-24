'''A library for machine learning metrics.

This library was born out of the need for out of core metrics. Most statistics
can be implemented with online algorithms with only a constant memory overhead.
That is to say that the entire dataset need not be in core memory. However
existing packages, like `sklearn.metrics`, typically implement these metrics as
Python functions which take the full dataset as inputs.

To aleviate this overhead, this library introduces an `Accumulator` interface.
Rather mapping datasets to metrics in a single function, accumulators recieve
batches of the dataset through an `accumulate` method and return the computed
metric through the `reduce` method.
'''

import abc


class Accumulator(metaclass=abc.ABCMeta):
    '''An abstract base class for the Accumulator interface.
    '''

    @abc.abstractmethod
    def accumulate(self, *batches):
        '''Accumulate some batch of data.

        Implementors may accept any number of batches, under the contract that
        the ith element of each batch all correspond to the same instance. This
        is important for things like classification metrics which typically
        recieve both a batch of predictions and a corresponding batch of ground
        truths.

        Implementors must not accept any additional arguments.

        Users should call this method on each batch of the data, without
        overlap. Every instance observed by the accumulator is considered
        distinct.
        '''
        pass

    @abc.abstractmethod
    def reduce(self):
        '''Return the accumulated metric, and reset to the initial state.

        Implementors must not accept any additional arguments.

        Users should call this method after all batches have been observed by
        the `accumulate` method. Once the accumulator is reduced, users may
        continue to use the accumulator as if no data has been observed.
        '''
        pass
