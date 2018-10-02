Loading data
==================================================

.. contents::
    :local:

Prerequisites
--------------------------------------------------

This guide will walk through loading NAM weather forecasts and preparing the data for training Scikit-learn or PyTorch models. This guide assumes you have an active conda environment with the appropriate dependencies. If not, see the :doc:`getting-started` guide.

There are two APIs for loading forecast data. The high-level API at :mod:`apollo.datasets.solar` provides everything you need to quickly access forecast data for training. It is built upon the low-level API at :mod:`apollo.datasets.loaders` which provides finer grain control and is ideal for data exploration.

The High-Level Data API
--------------------------------------------------

.. module:: apollo.datasets.solar

The primary interface to the high-level API is the :class:`SolarDataset` class. This class implements a simple sequence interface; once the dataset object is constructed, indexing into it will return a tuple of numpy arrays for a particular sample, one array per feature::

    >>> from apollo.datasets.solar import SolarDataset
    >>> dataset = SolarDataset()
    >>> dataset[0]
    # TODO: record output

This API is sufficient to be loaded through the PyTorch :class:`torch.utils.data.DataLoader`. The DataLoader implements the iterator protocol and provides niceties like batching and shuffling/sampling::

    >>> from torch.utils.data import DataLoader
    >>> loader = DataLoader(dataset, batch_size=4, shuffle=True)
    >>> next(iter(loader))
    # TODO: record output

Most Scikit-learn models expect their inputs to be a two dimension array with one row per sample and one column per feature, and expect their targets to be a one dimension array with one element per sample. The method :meth:`SolarDataset.tabular` will cast the dataset to a pair of arrays ``(x, y)`` satisfying these requirements. The arrays will be backed by the :mod:`dask` distributed system, which is used by the underlying implementation to lazily stream data from disk. In the case of Scikit-learn models, it may be beneficial when possible to collect the entire dataset into main memory by casting to a numpy array::

    >>> import numpy as np
    >>> x, y = dataset.tabular()
    >>> x = np.asarray(x)
    >>> y = np.asarray(y)
    >>> x.shape
    # TODO: record output
    >>> y.shape
    # TODO: record output

The :class:`SolarDataset` provides several preprocessing features. It makes it easy to select temporal, spatial, and feature subsets of the data; it can compute sine/cosine temporal features for time of day and time of year; it can generate sliding windows over the data; and it can standardize the data to center mean and unit variance. For more information, see the API docs.

The Low-Level Data API
--------------------------------------------------

.. module:: apollo.datasets.loaders

.. todo::
    **TODO**: Document the NAM loader.

.. todo::
    **TODO**: Document the GA Power loader. It must first be ported to the new
