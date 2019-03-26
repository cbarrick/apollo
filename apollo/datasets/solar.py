'''The primary dataset API in apollo.

The :mod:`apollo.datasets.solar` contains the primary dataset class
and helper functions for apollo. It handles the joining of NAM input
variables with Georgia Power targets along with various preprocessing.

The primary class, :class:`SolarDataset`, exposes the data as a
sequence of tuples, where each tuple is a single training instance, and
each element is a numpy array giving the data for a particular feature.
The target feature is always the final element. This format is
compatible with the PyTorch :class:`~torch.utils.data.DataLoader` API,
but note that the prefetching feature in PyTorch may interfear with the
underlying Dask API and cause deadlocks (as of August 2018).

The method :meth:`SolarDataset.tabular` casts the dataset to a pair of arrays.
It flattens the spatial and temporal dimensions into separate features. This is
most useful for Scikit-Learn estimators and XGBoost which do not (and cannot)
exploit the high dimension shape of the features.
'''

from copy import deepcopy
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial
import xarray as xr

import apollo.storage
from apollo.datasets import nam, ga_power


#: The latitude and longitude of the solar array.
#:
#: .. note::
#:     This is was taken from Google Maps as the lat/lon of the State
#:     Botanical Garden of Georgia, because that was the nearest I could find.
ATHENS_LATLON = (33.9052058, -83.382608)

# default targets loaded by SolarDataset
DEFAULT_TARGET = 'UGABPOA1IRR'
DEFAULT_TARGET_HOURS = tuple(range(1, 25))

#: The planar features of the NAM dataset,
#: i.e. those where the Z-axis has size 1.
PLANAR_FEATURES = (
    'PRES_SFC',
    'HGT_SFC',
    'HGT_TOA',
    'TMP_SFC',
    'VIS_SFC',
    'UGRD_TOA',
    'VGRD_TOA',
    'DSWRF_SFC',
    'DLWRF_SFC',
)


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
        array:
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
        data (xarray.Dataset):
            The dataset to slice, having dimension coordinates 'y' and 'x' and
            non-dimension coordinates 'lat' and 'lon' labeled by `(y, x)`.
        center ([lat, lon]):
            The latitude and longitude of the center.
        shape ([height, width]):
            The height and width of the selection in grid units.

    Returns:
        xarray.Dataset:
            The result of slicing data.
    '''
    lat, lon = data['lat'], data['lon']
    latlon = np.stack([lat, lon])
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
        data (xarray.Dataset):
            The dataset from which to extract features, having a dimension
            coordinate named 'reftime'.

    Returns:
        xarray.Dataset:
            A dataset with 4 data variables:
                - ``time_of_year_sin``
                - ``time_of_year_cos``
                - ``time_of_day_sin``
                - ``time_of_day_cos``
    '''
    reftime = data['reftime'].astype('float64')
    time_of_year = reftime / 3.1536e+16  # convert from ns to year
    time_of_day = reftime / 8.64e+13  # convert from ns to day

    return xr.Dataset({
        'time_of_year_sin': np.sin(time_of_year * 2 * np.pi),
        'time_of_year_cos': np.cos(time_of_year * 2 * np.pi),
        'time_of_day_sin': np.sin(time_of_day * 2 * np.pi),
        'time_of_day_cos': np.cos(time_of_day * 2 * np.pi),
    })


def window_reftime(base, lag):
    '''Creates a sliding window over the reftime axis.

    This operation works by duplicating and shifting the data variables along
    the reftime axis in 6h increments.

    Arguments:
        base (xarray.Dataset):
            The dataset to window.
        lag (int):
            The size of the window.

    Returns:
        xarray.Dataset:
            The windowed dataset.
    '''
    datasets = [base]
    base_names = list(base.data_vars)

    new_names = {f'{name}':f'{name}_0' for name in base_names}
    data = base.isel(forecast=0).rename(new_names)

    timedelta = np.timedelta64(6, 'h')

    for i in range(lag):
        data = data.copy()
        data['reftime'] = data['reftime'] + timedelta
        new_names = {f'{name}_{i}':f'{name}_{i+1}' for name in base_names}
        data = data.rename(new_names)
        datasets.append(data)

    return xr.merge(datasets, join='inner')


def load_targets(target, start, stop, target_hours):
    '''Load the target variable.

    Arguments:
        target (str):
            The name of a variable in the GA Power dataset to include as a target.
        start (timestamp):
            The timestamp of the first reftime.
        stop (timestamp):
            The timestamp of the final reftime.
        target_hours (Iterable[int]):
            The hour offsets of the target in the reftime dimension.

    Returns:
        target_data (xr.DataArray):
            A data array for the target variable.
            Its shape is ``(reftime, target_hour)``.
    '''
    # Normalize the target_hours to a list,
    target_hours = list(target_hours)

    # Load the raw target data, corresponding to a 0 hour prediction.
    # Note that 'target_hour' is a non-dimension coordinate. We later
    # use `DataArray.expan_dims` to promote it to a dimension.
    target_data_raw = ga_power.open_sqlite(target, start=start, stop=stop)
    target_data_raw['target_hour'] = 0

    # Create a DataArray for each target hour.
    # They all have the same name (the value of `target`),
    # and the same dimensions ('reftime' and 'target_hour')
    # but the values of the dimensions may be different.
    target_data_arrays = []
    for hour in target_hours:
        # The deep copy in xarray does not copy coordinates or attributes, so we do it manually.
        # See https://github.com/pydata/xarray/issues/1463
        # and https://github.com/cbarrick/apollo/issues/39
        x = target_data_raw[target].copy(deep=True)
        for coord in x.coords:
            x.coords[coord].data = np.copy(x.coords[coord].data)
        x.attrs = deepcopy(x.attrs)
        for attr in x.attrs:
            x.attrs[attr] = deepcopy(x.attrs[attr])

        # lag target values
        x['reftime'] -= np.timedelta64(int(hour), 'h')
        x['target_hour'] = hour
        x = x.expand_dims('target_hour', 1)
        target_data_arrays.append(x)

    # Concat the target data arrays together along a new dim,
    # Drop N/A values along the reftime dimension to make it an inner join.
    target_data = xr.concat(target_data_arrays, dim='target_hour')
    target_data = target_data.dropna('reftime')
    return target_data


class SolarDataset:
    '''A high level interface for the solar prediction dataset.

    This class unifies the NAM-NMM and GA Power datasets and provides many
    functionalities including feature and geographic subsetting, sliding
    window, time of day/year extraction, and standardization.

    This class exposes a PyTorch Dataset interface (i.e. `__len__` and
    `__getitem__`). Each row is a tuple with one element per feature. If a
    target variable is used, it will always be the final column.

    The dataset can be dumped to a tabular array, useful for experiments
    with Scikit-learn.

    Attributes:
        xrds (xarray.Dataset):
            The underlying xarray dataset.
        target (str or None):
            The name of the target variable.
        labels (tuple of str):
            Labels for each feature column.
        standardized (bool):
            A flag indicating if the data is standardized.
        mean (xarray.Dataset or float):
            If the data is standardized, a dataset containing the mean
            values used to center the data variables, or 0 otherwise.
        std (xarray.Dataset or float):
            If the data is standardized, a dataset containing the standard
            deviations used to scale the data variables, or 1 otherwise.
        target_hours (Tuple[int]):
            The target hours of the labels.
    '''

    def __init__(self, start='2017-01-01 00:00', stop='2017-12-31 18:00', *,
                 feature_subset=PLANAR_FEATURES, temporal_features=True,
                 geo_shape=(3, 3), center=ATHENS_LATLON, lag=0, forecast=36,
                 target=DEFAULT_TARGET, target_hours=DEFAULT_TARGET_HOURS,
                 standardize=True):
        '''Initialize a SolarDataset

        Arguments:
            start (timestamp):
                The timestamp of the first reftime.
            stop (timestamp):
                The timestamp of the final reftime.
            feature_subset (set of str or None):
                The set of features to select. If None, all features are used.
                The default `PLANAR_FEATURES` is a selection of features with
                trivial z-axes.
            temporal_features (bool):
                If true, extend with additional temporal features for time of
                day and time of year.
            geo_shape (Tuple[int, int] or None):
                If given, the y and x axes are sliced to this shape, in grid
                units (roughly 12km). The default `ATHENS_LATLON` is the rough
                location of the solar farm which collects the target data.
            center (Tuple[float, float]):
                The latitude and longitude of the center geographic slice.
                This only applies when ``geo_shape`` is not None.
            lag (int):
                If greater than 0, create a sliding window over the reftime
                axis for data variables at the 0-hour forecast.
            forecast (int):
                The maximum forecast hour to include.
            target (str or None):
                The name of a variable in the GA Power dataset to include as a
                target. If a target is given the year of the start and stop
                timestamps must be the same (this can be improved).
            target_hours (int or Iterable[int]):
                The hour offsets of the target in the reftime dimension.
                This argument is ignored if ``target`` is None.
            standardize (bool or Tuple[xarray.Dataset, xarray.Dataset]):
                If true, standardize the data to center mean and unit standard
                deviation. If a tuple of datasets (or floats), standardize using
                use the given ``(mean, std)``. Do nothing if false. Note that
                the target column is never standardized.
        '''
        assert 0 <= lag

        start = pd.Timestamp(start) - pd.Timedelta(6, 'h') * lag
        stop = pd.Timestamp(stop)
        data = nam.open_range(start, stop)

        if feature_subset:
            data = data[list(feature_subset)]

        if geo_shape:
            data = slice_xy(data, center, geo_shape)

        if forecast is not None:
            data = data.isel(forecast=slice(0, forecast+1))

        if bool(standardize) is False:
            mean = 0.0
            std = 1.0
        elif standardize is True:
            mean = data.mean()
            std = data.std()
            data = (data - mean) / std
        else:
            mean, std = standardize
            data = (data - mean) / std

        if 0 < lag:
            data = window_reftime(data, lag)

        if temporal_features:
            temporal_data = extract_temporal_features(data)
            data = xr.merge([data, temporal_data])

        if target:
            try:
                target_hours = tuple(target_hours)
            except TypeError:
                target_hours = (target_hours,)
            target_data = load_targets(target, start, stop, target_hours)
            data = xr.merge([data, target_data], join='inner')
        else:
            target_hours = ()

        self.xrds = data.persist()
        self.target = target or None
        self.standardized = bool(standardize)
        self.mean = mean
        self.std = std
        self.target_hours = target_hours

    def __len__(self):
        return len(self.xrds['reftime'])

    def __getitem__(self, index):
        data = self.xrds.isel(reftime=index)
        arrays = data.data_vars.values()
        arrays = (np.asarray(a) for a in arrays)

        if self.target:
            target = data[self.target]
            target = np.asarray(target)
            return (*arrays, target)
        else:
            return (*arrays,)

    @property
    def labels(self):
        '''The labels of each feature column.
        '''
        names = tuple(self.xrds.data_vars)
        if self.target:
            return (*names, self.target)
        else:
            return (*names,)

    @property
    def shape(self):
        '''The shape of each feature column.
        '''
        names = self.labels
        data = self.xrds.isel(reftime=0)
        return tuple(data[name].shape for name in names)

    def tabular(self):
        '''Return a tabular version of the dataset as dask arrays.

        Many features of this dataset include spatial and temporal dimensions.
        This method flattens and concatenates all features into a single vector
        per instance.

        The return values are dask arrays streamed from distributed memory. For
        Scikit-learn experiments, you likely want to cast these to numpy arrays
        in main memory using `np.asarray`.

        Returns:
            x (array of shape (n,m)):
                The input features, where `n` is the number of instances and
                `m` is the size of the flattened feature vector.
            y (array of shape (n,)):
                The target array. This is only returned if a target is given
                to the constructor.
        '''
        n = len(self.xrds['reftime'])

        if self.target:
            x = self.xrds.drop(self.target)
        else:
            x = self.xrds

        x = x.data_vars.values()
        x = (a.data for a in x)
        x = (a.reshape(n, -1) for a in x)
        x = da.concatenate(list(x), axis=1)

        if self.target:
            y = self.xrds[self.target]
            y = y.data
            return x, y
        else:
            return x
