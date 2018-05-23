from pathlib import Path

import numpy as np
import scipy as sp
import scipy.spatial
import xarray as xr

import dask
from dask import array as da
from dask.distributed import Client

import torch
from torch.utils.data import Dataset as TorchDataset

from apollo.datasets import nam, ga_power


# The latitude and longitude of the solar array.
# NOTE: This is was taken from Google Maps as the lat/lon of the State
# Botanical Garden of Georgia, because that was the nearest I could find.
ATHENS_LATLON = [33.9052058, -83.382608]


# The planar features of the NAM dataset,
# i.e. those where the Z-axis has size 1.
PLANAR_FEATURES = [
    'PRES_SFC',
    'HGT_SFC',
    'HGT_TOA',
    'TMP_SFC',
    'VIS_SFC',
    'UGRD_TOA',
    'VGRD_TOA',
    'DSWRF_SFC',
    'DLWRF_SFC',
]


def find_nearest(data, point):
    '''Find the index into ``data`` nearest to ``point``.

    The point and data share their outer dimension. In other words, if ``data``
    has axes ``(a, b, c)`` then ``point`` should have a single axis ``(a,)``.
    The value returned indexes into axes ``(b, c)``.

    Arguments:
        data (array):
            The data to search.
        point (array):
            Search for the index nearest to this point.

    Returns:
        The unraveled indices into `data` of the cells nearest to `points`.
    '''
    data = np.require(data)
    point = np.require(point)
    (n, *shape) = data.shape
    assert len(data) == len(point)

    data = data.reshape(n, -1).T  # cdist wants data flattened and transposed
    points = point.reshape(1, n)  # cdist wants a batch of points
    distance = sp.spatial.distance.cdist(points, data)
    idx = distance.argmin(axis=1)[0]  # we only have one point
    return np.unravel_index(idx, shape)


def slice_xy(data, center, shape):
    '''Slice a dataset in the x and y dimensions.

    Arguments:
        data (xr.Dataset):
            The dataset to slice, having dimension coordinates 'y' and 'x' and
            non-dimension coordinates 'lat' and 'lon' labeled by `(y, x)`.
        center ([lat, lon]):
            The latitude and longitude of the center.
        shape ([height, width]):
            The height and width of the selection in grid units.

    Returns:
        subset (xr.Dataset):
            The result of slicing data.
    '''
    latlon = np.stack([data['lat'], data['lon']])
    i, j = find_nearest(latlon, center)
    h, w = shape
    top = i - int(np.ceil(h/2)) + 1
    bottom = i + int(np.floor(h/2)) + 1
    left = j - int(np.ceil(w/2)) + 1
    right = j + int(np.floor(w/2)) + 1
    slice_y = slice(top, bottom)
    slice_x = slice(left, right)
    return data.isel(y=slice_y, x=slice_x)


def extract_temporal_features(data):
    '''Extract temporal features from a dataset.

    Arguments:
        data (xr.Dataset):
            The dataset from which to extract features, having a dimension
            coordinate named 'reftime'.

    Returns:
        time_data (xr.Dataset):
            A dataset with 4 data variables:
                - ``time_of_year_sin``
                - ``time_of_year_cos``
                - ``time_of_day_sin``
                - ``time_of_day_cos``
    '''
    reftime = data['reftime'].astype('float64')

    time_of_year = reftime / 3.1536e+16  # convert from ns to year
    time_of_year_sin = np.sin(time_of_year * 2 * np.pi)
    time_of_year_cos = np.cos(time_of_year * 2 * np.pi)

    time_of_day = reftime / 8.64e+13  # convert from ns to day
    time_of_day_sin = np.sin(time_of_day * 2 * np.pi)
    time_of_day_cos = np.cos(time_of_day * 2 * np.pi)

    return xr.Dataset({
        'reftime': reftime,
        'time_of_year_sin': time_of_year_sin,
        'time_of_year_cos': time_of_year_cos,
        'time_of_day_sin': time_of_day_sin,
        'time_of_day_cos': time_of_day_cos,
    })


def create_window(base, window_size):
    '''Creates a sliding window over the reftime axis.

    Arguments:
        base (xr.Dataset):
            The dataset to window.
        window_size (int):
            The size of the window.

    Returns:
        windowed (xr.Dataset):
            The input dataset extended with a silding window.
    '''
    datasets = [base]
    base_names = list(base.data_vars)

    new_names = {f'{name}':f'{name}_0' for name in base_names}
    data = base.isel(forecast=0).rename(new_names)

    timedelta = np.timedelta64(6, 'h')

    for i in range(window_size - 1):
        data = data.copy()
        data['reftime'] = data['reftime'] + timedelta
        new_names = {f'{name}_{i}':f'{name}_{i+1}' for name in base_names}
        data = data.rename(new_names)
        datasets.append(data)

    return xr.merge(datasets, join='inner')


class SolarDataset(TorchDataset):
    def __init__(self, start='2017-01-01 00:00', stop='2017-12-31 18:00', *,
            feature_subset=PLANAR_FEATURES, temporal_features=True,
            center=ATHENS_LATLON, geo_shape=(3, 3), window=1,
            target='UGA-C-POA-1-IRR', target_hour=24,
            standardize=True, cache_dir='./data'):

        assert 0 < window

        # Create local Dask cluster and connect.
        # This is not required, but doing so adds useful debugging features.
        self.client = Client()

        cache_dir = Path(cache_dir)
        nam_cache = cache_dir / 'NAM-NMM'
        target_cache = cache_dir / 'GA-POWER'

        data = nam.open_range(start, stop, cache_dir=nam_cache)

        if feature_subset:
            data = data[feature_subset]

        if geo_shape:
            data = slice_xy(data, center, geo_shape)

        if standardize:
            mean = data.mean()
            std = data.std()
            data = (data - mean) / std

        if 1 < window:
            data = create_window(data, window)

        if temporal_features:
            temporal_data = extract_temporal_features(data)
            data = xr.merge([data, temporal_data])

        if target:
            # When using targets, the start and stop year must be the same.
            # This is because the targets are broken out by year and the loader
            # only loads one group. This can be improved...
            year = np.datetime64(start, 'Y')
            stop_year = np.datetime64(stop, 'Y')
            assert year == stop_year, "start and stop must be same year"

            target_data = ga_power.open_mb007(target, data_dir=target_cache, group=year)
            target_data['reftime'] -= np.timedelta64(target_hour, 'h')
            data = xr.merge([data, target_data], join='inner')
            data = data.set_coords(target)  # NOTE: the target is a coordinate, not data

        self.dataset = data.persist()
        self.target = target

    def __len__(self):
        return len(self.dataset['reftime'])

    def __getitem__(self, index):
        dataset = self.dataset.isel(reftime=index)
        arrays = dataset.data_vars.values()
        arrays = (np.as_array(a) for a in arrays)

        if self.target:
            target = dataset[self.target]
            target = np.asarray(target)
            return (*arrays, target)
        else:
            return (*arrays,)

    def labels(self):
        '''Get the labels of the columns.
        '''
        names = tuple(self.dataset.data_vars)
        if self.target:
            return (*names, self.target)
        else:
            return (*names,)

    def tabular(self):
        '''Get a tabular version of the dataset as a dask array(s).

        Returns:
            x (array of shape (n,m)):
                The input features, where `n` is the number of instances and
                `m` is the number of columns after flattening the features
            y (array of shape (n,)):
                The target array. This is only returned if a target is given
                to the constructor.
        '''
        n = len(self.dataset['reftime'])
        x = self.dataset.data_vars.values()
        x = (a.data for a in x)
        x = (a.reshape(n, -1) for a in x)
        x = np.concatenate(list(x), axis=1)

        if self.target:
            y = self.dataset[self.target]
            y = y.data
            assert y.shape == (n,)
            return x, y
        else:
            return x
