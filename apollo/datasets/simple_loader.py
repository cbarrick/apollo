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
    inputs = nam.open_range(start, stop, cache_dir=cache_dir + '/NAM-NMM')

    # load data from solar farm
    targets = ga_power.open_mb007(target, data_dir=cache_dir + '/GA-POWER')

    # inner join with nam data to eliminate missing times
    targets['reftime'] -= np.timedelta64(target_hour, 'h')
    data = xr.merge([inputs, targets], join='inner')

    # extract features for time-of-day and time-of-year
    timedelta = data['reftime']
    timedelta = timedelta - timedelta[0]
    timedelta = timedelta.astype('float64')

    time_of_year = timedelta / 3.1536e+16  # convert from ns to year
    time_of_year_sin = np.sin(time_of_year * 2 * np.pi)
    time_of_year_cos = np.cos(time_of_year * 2 * np.pi)
    data['time_of_year_sin'] = time_of_year_sin
    data['time_of_year_cos'] = time_of_year_cos

    time_of_day = timedelta / 8.64e+13  # convert from ns to day
    time_of_day_sin = np.sin(time_of_day * 2 * np.pi)
    time_of_day_cos = np.cos(time_of_day * 2 * np.pi)
    data['time_of_day_sin'] = time_of_day_sin
    data['time_of_day_cos'] = time_of_day_cos

    # Find the index of the cell nearest to the given lat and lon.
    # I've been using the coordinates of the Botanical Gardens since I don't know the exact location of the solar farm.
    # The `nam.find_nearest` function helps some, but this could still be cleaned up.
    latlon = [33.9052058, -83.382608]
    lat = data['lat'].data
    lon = data['lon'].data
    (pos_y, pos_x) = nam.find_nearest(np.stack([lat, lon]), latlon)[0]

    # Slice out the region we want.
    slice_y = slice(pos_y - 1, pos_y + 2)
    slice_x = slice(pos_x - 1, pos_x + 2)
    region = data[{'y': slice_y, 'x': slice_x}]

    # Select the desired input variables.
    # The syntax `dataset[['var1', 'var2', ...]]` returns a reduced dataset.
    # Select the desired input variables.
    if desired_attributes == 'all':
        x = region
    else:
        x = region[desired_attributes]

    # Extract the underlying arrays.
    x = [arr.data for arr in x.data_vars.values()]  # The underlying array may be dask or numpy.
    x = da.concatenate(x, axis=2)  # Stack along the z axis.

    # scikit-learn wants tabular data.
    x = x.reshape(len(x), -1)

    # Stack on the time_of_day and time_of_year features.
    times = region[['time_of_day_sin', 'time_of_day_cos', 'time_of_year_sin', 'time_of_year_cos']]
    times = [arr.data for arr in times.data_vars.values()]
    times = da.stack(times, axis=1)
    x = da.concatenate([x, times], axis=1)

    # Select the target, and extract the underlying array.
    y = data['UGA-C-POA-1-IRR']
    y = y.data

    # ensure the data are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    return x, y
