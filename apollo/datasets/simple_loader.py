"""
The simple_loader module provides high-level functions for loading cached data into in-memory numpy arrays

The loader expects a `cache_dir` argument, which specifies the directory where the NAM and Georgia Power data can be
found.  NAM forecasts generated from apollo.datasets.open should be located in the <cache_dir>/NAM-NMM directory.
There should be a single log file
"""

import numpy as np
import dask.array as da
import xarray as xr

from apollo.datasets import nam, ga_power


def load(start='2017-01-01 00:00', stop='2017-12-31 18:00', target_hour=24, target_var='UGA-C-POA-1-IRR',
         cache_dir='/mnt/data6tb/chris/data', standardize=True, desired_attributes='surface', grid_size=3):
    """
    Loads a dataset from cached grib files into a numpy array
    This function assumes that cache_dir contains two subdirectories: NAM-NMM containing NAM forecast summary files
    and GA-POWER, containing an uncompressed log file with the ML targets (see apollo/bin/generate_targets.sh)

    :param start: anything accepted by numpy's np.datetime64 constructor
        The reference time of the first data point to be selected
        DEFAULT: 2017-01-01 00:00
    :param stop: anything accepted by numpy's np.datetime64 constructor
        The reference time of the last data point to be selected
        DEFAULT: 2017-12-31 18:00
    :param target_hour: integer in [1, 36]
        The hour you are targeting for solar radiation prediction
        DEFAULT: 24
    :param target_var: string
        The name of variable from the solar farm logs that we are trying to predict.
        DEFAULT: 'UGA-C-POA-1-IRR', a ventilated pyranometer on a fixed solar array
    :param cache_dir: string
        The local directory where the data resides on disk.  Should have subfolders 'NAM-NMM' containing NAM forecasts
        and 'GA-POWER' containing the data from the solar farm.
        The default cache directory is where the data is currently stored on aigpc3.
        DEFAULT: /mnt/data6tb/chris/data
    :param standardize: boolean
        Should the data be standardized during loading?
        DEFAULT: True
    :param desired_attributes: array<str> or keyword 'all' or keyword 'surface'
        Array containing the names of the data variables to select, or the keyword 'all' to select all data variables,
        or the keyword 'surface' to select surface variables.
        DEFAULT: 'surface'
    :param grid_size: odd integer >= 1
        The size of the spatial grid from which features will be selected.
        Features will be included from all cells in a (grid_size x grid_size) spatial grid centered on the cell where
        the solar array resides.
        Values will be rounded up to the nearest odd number >= 1
        DEFAULT: 3

    :return: (X, y) where X is an n x m np.array containing the non-target attributes,
             and y is an n x 1 np.array containing the target values
    """

    # ensure the user hasn't requested data for two different years
    assert np.datetime64(start).astype(object).year == np.datetime64(stop).astype(object).year, \
        "Loading across different years is not yet supported."
    year = np.datetime64(start).astype(object).year

    # open weather forecast data
    nam2017 = nam.open_range(start, stop, cache_dir=cache_dir + '/NAM-NMM')

    # open readings from the targeted solar array
    targets = ga_power.open_mb007(target_var, data_dir=cache_dir + '/GA-POWER', group=year)

    # pair input forecasts with radiation observations n hours in the future by subtracting n hours from the
    # target reftimes and performing an inner join with the forecast data
    targets['reftime'] -= np.timedelta64(target_hour, 'h')
    full_data = xr.merge([nam2017, targets], join='inner')

    # Find the index of the cell nearest to the given lat and lon.
    # I've been using the coordinates of the Botanical Gardens since I don't know the exact location of the solar farm.
    # The `nam.find_nearest` function helps some, but this could still be cleaned up.
    latlon = [33.9052058, -83.382608]
    lat = full_data['lat'].data
    lon = full_data['lon'].data
    (pos_y, pos_x) = nam.find_nearest(np.stack([lat, lon]), latlon)[0]

    # Slice out the region we want.
    grid_size = max(int(grid_size), 1)  # ensure grid_size is an integer >= 1
    if grid_size % 2 != 1:
        grid_size += 1

    offset = (grid_size - 1) // 2  # the number of cells that the grid extends in each direction
    slice_y = slice(pos_y - offset, pos_y + offset + 1)
    slice_x = slice(pos_x - offset, pos_x + offset + 1)
    data = full_data[{'y': slice_y, 'x': slice_x}]

    # for some algorithms, the data should be centered around the mean and normalized to unit variance
    if standardize:
        standards = {}
        for name, var in data.data_vars.items():
            mean = var.mean()
            std = var.std()
            standards[name] = {'mean': mean, 'std': std}
            data[name] = (var - mean) / std

    # extract periodic features for the time of day and time of year
    timedelta = data['reftime']
    timedelta = timedelta - timedelta[0]
    timedelta = timedelta.astype('float64')

    time_of_year = timedelta / 3.1536e+16  # convert from ns to year
    time_of_year_sin = np.sin(time_of_year * 2 * np.pi)
    time_of_year_cos = np.cos(time_of_year * 2 * np.pi)

    time_of_day = timedelta / 8.64e+13  # convert from ns to day
    time_of_day_sin = np.sin(time_of_day * 2 * np.pi)
    time_of_day_cos = np.cos(time_of_day * 2 * np.pi)

    time_features = [time_of_year_sin, time_of_year_cos, time_of_day_sin, time_of_day_cos]
    time_features = da.stack(time_features, axis=1)  # stacks time features as columns

    if desired_attributes == 'all':
        planar_features = data
    elif desired_attributes == 'surface':
        planar_features = data[[
            'PRES_SFC',
            'HGT_SFC',
            'HGT_TOA',
            'TMP_SFC',
            'VIS_SFC',
            'UGRD_TOA',
            'VGRD_TOA',
            'DSWRF_SFC',
            'DLWRF_SFC',
        ]]
    else:
        planar_features = data[desired_attributes]

    # Extract the underlying arrays and stack
    # The len test ensures that we only stack data variables with a third axis (this should nearly always be the z-dim)
    planar_features = [arr.data for arr in planar_features.data_vars.values() if len(arr.dims) >= 3]
    planar_features = da.concatenate(planar_features, axis=2)

    # scikit-learn models expect tabular data
    n = len(planar_features)
    planar_tabular = planar_features.reshape(n, -1)
    x = np.concatenate([planar_tabular, time_features], axis=1)
    y = data[target_var].data

    return x, y
