"""
The simple_loader module provides high-level functions for loading datasets into in-memory numpy arrays
This format is useful for simple models (a la scikit-learn)
"""

import numpy as np
import dask.array as da
import xarray as xr

from apollo.datasets import nam, ga_power


def load(target_hour=24, desired_attributes='all', start='2017-01-01 00:00', stop='2017-12-31 18:00',
         target='UGA-C-POA-1-IRR', cache_dir='/mnt/data6tb/chris/data'):
    """
    Loads a dataset from cached grib files into a numpy array

    :param target_hour: integer in [1, 36]
        The hour you are targeting for solar radiation prediction
    :param desired_attributes: array<str> or keyword 'all'
        Array containing the names of the data variables to select, or the keyword 'all' to select all data variables
    :param start: anything accepted by numpy's np.datetime64 constructor
        The reference time of the first data point to be selected
    :param stop: anything accepted by numpy's np.datetime64 constructor
        The reference time of the last data point to be selected
    :param target: string
        The name of value from the solar farm data that we are trying to predict
    :param cache_dir: string
        The local directory where the data resides on disk.  Should have subfolders 'NAM-NMM' containing NAM forecasts
        and 'GA-POWER' containing the data from the solar farm

    :return: (X, y) where X is a np.array containing the non-target attributes, and y is a np.array containing the target values
    """

    # load nam data
    inputs = nam.open_range('2017-01-01 00:00', '2017-12-31 18:00', cache_dir=cache_dir + '/NAM-NMM')
    # load data from solar farm
    targets = ga_power.open_mb007(target, data_dir=cache_dir + '/GA-POWER')

    # inner join with nam data to eliminate missing times
    targets['reftime'] -= np.timedelta64(target_hour, 'h')
    data = inputs.merge(targets, join='inner')

    # Find the index of the cell nearest to the given lat and lon.
    # I've been using the coordinates of the Botanical Gardens since I don't know the exact location of the solar farm.
    # The `nam.find_nearest` function helps some, but this could still be cleaned up.
    latlon = [33.9052058, -83.382608]
    lat = data['lat'].data
    lon = data['lon'].data
    (pos_y, pos_x) = nam.find_nearest(np.stack([lat, lon]), latlon)[0]

    # Get slice out the region we want.
    slice_y = slice(pos_y - 1, pos_y + 2)
    slice_x = slice(pos_x - 1, pos_x + 2)
    data = data[{'y': slice_y, 'x': slice_x}]

    # Select the desired input variables.
    # The syntax `dataset[['var1', 'var2', ...]]` returns a reduced dataset.
    if desired_attributes == 'all':
        x = data
    else:
        x = data[desired_attributes]

    # Pull out the raw arrays.
    # Note `xr.DataArray.data` returns the underlying array, which may be a Dask or numpy array.
    x = list(x.data_vars.values())  # Convert to a list of `xr.DataArray`.
    x = [arr.data for arr in x]  # Convert to a list of Dask arrays.
    x = da.concatenate(x, axis=2)  # Stack along the z axis.
    x = x.reshape(-1, 37 * 9 * 3 * 3)  # Make the array tabular.
    x = x.compute()  # Collect into memory.

    # Select the desired target.
    y = data[target]  # The syntax `dataset['var']` returns a single `xr.DataArray`.
    y = y.data  # Convert the xarray DataArray into a numpy array.

    return x, y
