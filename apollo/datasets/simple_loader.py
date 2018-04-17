"""
The simple_loader module provides high-level functions for loading cached data into in-memory numpy arrays
"""

import numpy as np
import dask.array as da
import xarray as xr

from apollo.datasets import nam, ga_power


def load(start='2017-01-01 00:00', stop='2017-12-31 18:00', desired_attributes='surface', grid_size=3,
         cache_dir='/mnt/data6tb/chris/data', standardize=True, target_var='UGA-C-POA-1-IRR', target_hour=24):
    """Loads a dataset from on-disk GRIB files into an in-memory numpy array

    By default, loads weather forecast attributes from local grib files and joins them with the target values found in
    logs from Georgia Power.  Data should be cached (downloaded) in the cache_dir.
    NAM forecasts generated from apollo.datasets.open should be located in the <cache_dir>/NAM-NMM directory.
    There should be a single uncompressed log file located in the <cache_dir>/GA-POWER directory that has
    target readings from the solar array.

    This function can also be used to load plain NAM data (with no targets) by setting the `target_var` to None.

    All parameters are optional.

    Args:
        start (str): Specifies the reference time of the first data point to be selected.
            Any string accepted by the np.datetime64 constructor is acceptable.
        stop (str): Specifies the reference time of the last data point to be selected.  Any string accepted by
            the np.datetime64 constructor is acceptable.
        cache_dir (str): The local directory where the data can be found.  Should have subfolders 'NAM-NMM'
            containing NAM forecasts and 'GA-POWER' containing the data from the solar farm.
        standardize (bool): Should the data be standardized?
        desired_attributes (array or str): One-dimensional numpy array with the names of the data variables to select,
            or the keyword 'all' to select all data variables, or the keyword 'surface' to select surface variables.
        grid_size (int): The size of the spatial grid from which features will be selected.
            This parameter will be rounded up to the nearest odd integer, and a grid of shape (grid_size, grid_size)
            will be selected.
        target_var (str): Name of the reading from Georgia Power logs which will be used as the target.
        target_hour (int): Integer in the range [1, 36] specifying the target prediction hour

    Returns:
        A tuple (x, y), where where x is an n x m np.array containing the non-target attributes,
        and y is an n x 1 np.array containing the target values.

        If the target_var is None, then y will be None.

    """

    # ensure the user hasn't requested data for two different years
    assert np.datetime64(start).astype(object).year == np.datetime64(stop).astype(object).year, \
        "Loading across different years is not yet supported."
    year = np.datetime64(start).astype(object).year

    # open weather forecast data
    nam_data = nam.open_range(start, stop, cache_dir=cache_dir + '/NAM-NMM')

    if target_var is not None:
        # open readings from the targeted solar array
        targets = ga_power.open_mb007(target_var, data_dir=cache_dir + '/GA-POWER', group=year)

        # pair input forecasts with radiation observations n hours in the future by subtracting n hours from the
        # target reftimes and performing an inner join with the forecast data
        targets['reftime'] -= np.timedelta64(target_hour, 'h')
        full_data = xr.merge([nam_data, targets], join='inner')
    else:
        full_data = nam_data

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
            if name != target_var and name != 'reftime':
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

    # sort the data by reftime
    data.sortby('reftime')

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
    if target_var is not None:
        y = data[target_var].data
    else:
        y = None

    return x, y
