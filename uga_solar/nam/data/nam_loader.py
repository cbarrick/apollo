#!/usr/bin/env python3
'''A NAM dataset downloader and DAO.

> The North American Mesoscale Forecast System (NAM) is one of the
> major weather models run by the National Centers for Environmental
> Prediction (NCEP) for producing weather forecasts. Dozens of weather
> parameters are available from the NAM grids, from temperature and
> precipitation to lightning and turbulent kinetic energy.

> As of June 20, 2006, the NAM model has been running with a non-
> hydrostatic version of the Weather Research and Forecasting (WRF)
> model at its core. This version of the NAM is also known as the NAM
> Non-hydrostatic Mesoscale Model (NAM-NMM).

Most users will be interested in the `load_nam` function which loads
the data for a single NAM-NMM run at a particular reference time.
The actual data loading logic is encapsulated in the `NAMLoader` class.

The data loading logic works like this:

1. If the forecast exists in the cache, load it.
2. Otherwise download the GRIBs for the forecast.
3. Extract the data from the GRIBs and reconstruct the z and time axes.
4. Cache the forecast as a netCDF and delete the GRIBs.

The dataset is returned as an `xarray.Dataset`, and each variable has
exactly five dimensions: reftime, forecast, z, y, and x. The z-axis for
each variable has a different name depending on the type of index
measuring the axis, e.g. `heightAboveGround` for height above the
surface in meters or `isobaricInhPa` for isobaric layers. The names of
the variables follow the pattern `FEATURE_LAYER` where `FEATURE` is a
short identifier for the feature being measured and `LAYER` is the type
of z-axis used by the variable, e.g. `t_isobaricInhPa` for the
temperature at the isobaric layers.

This module exposes several globals containing general metadata about
the NAM dataset.

This module also exposes a CLI to maintain the cache.
'''

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from itertools import groupby, repeat
from pathlib import Path
from time import sleep
import logging
import re

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial.distance
import pygrib
import requests
import xarray as xr


# Module level logger
logger = logging.getLogger(__name__)


# URLs of remote grib files.
# PROD_URL typically has the most recent 7 days.
# ARCHIVE_URL typically has the most recent 11 months, about 1 week behind.
PROD_URL = 'http://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.{ref.year:04d}{ref.month:02d}{ref.day:02d}/nam.t{ref.hour:02d}z.awphys{forecast:02d}.tm00.grib2'
ARCHIVE_URL_1 = 'https://nomads.ncdc.noaa.gov/data/meso-eta-hi/{ref.year:04d}{ref.month:02d}/{ref.year:04d}{ref.month:02d}{ref.day:02d}/nam_218_{ref.year:04d}{ref.month:02d}{ref.day:02d}_{ref.hour:02d}00_{forecast:03d}.grb'
ARCHIVE_URL_2 = 'https://nomads.ncdc.noaa.gov/data/meso-eta-hi/{ref.year:04d}{ref.month:02d}/{ref.year:04d}{ref.month:02d}{ref.day:02d}/nam_218_{ref.year:04d}{ref.month:02d}{ref.day:02d}_{ref.hour:02d}00_{forecast:03d}.grb2'


# The full forecast period of the NAM-NMM dataset: 0h to 36h by 1h and 36h to 84h by 3h
# The forecast period we work with: 0h to 36h by 1h
FULL_FORECAST_PERIOD = tuple(range(36)) + tuple(range(36, 85, 3))
FORECAST_PERIOD = FULL_FORECAST_PERIOD[:37]


# The projection of the NAM-NMM dataset as a `cartopy.crs.CRS`.
# The projection is officially called called "Grid 218" by NOAA.
# http://www.nco.ncep.noaa.gov/pmb/docs/on388/tableb.html#GRID218
NAM218_PROJ = ccrs.LambertConformal(
    central_latitude=25,
    central_longitude=265,
    standard_parallels=(25, 25),

    # The default cartopy globe is WGS 84, but
    # NAM assumes a spherical globe with radius 6,371.229 km
    globe=ccrs.Globe(ellipse=None, semimajor_axis=6371229, semiminor_axis=6371229),
)


# The projection of the NAM-NMM dataset as a CF convention grid mapping.
# This is stored in the netCDF files when they are converted.
# The projection is officially called called "Grid 218" by NOAA.
# http://www.nco.ncep.noaa.gov/pmb/docs/on388/tableb.html#GRID218
NAM218_PROJ_CF = {
    'grid_mapping_name': 'lambert_conformal_conic',
    'latitude_of_projection_origin': 25.0,
    'longitude_of_central_meridian': 265.0,
    'standard_parallel': 25.0,
    'earth_radius': 6371229.0,
}


# The geographic subset being used, given as grid indices in (y, x) order.
# This subset is centered at Macon, GA and covers all of Georgia
# and much of the neighboring states.
GEO_SUBSET = (slice(93, 194, None), slice(385, 486, None))


# The subset of features being used.
FEATURE_SUBSET = ['dlwrf', 'dswrf', 'pres', 'vis', 'tcc', 't', 'r', 'u', 'v', 'w']


# A description of features in the NAM-NMM dataset.
# Maps feature abreviations to their descriptions.
FEATURES = {
    '10u':    '10 metre U wind component',
    '10v':    '10 metre V wind component',
    '2d':     '2 metre dewpoint temperature',
    '2r':     'Surface air relative humidity',
    '2t':     '2 metre temperature',
    '4lftx':  'Best (4-layer) lifted index',
    'absv':   'Absolute vorticity',
    'acpcp':  'Convective precipitation (water)',
    'al':     'Albedo',
    'bmixl':  'Blackadar mixing length scale',
    'cape':   'Convective available potential energy',
    'cd':     'Drag coefficient',
    'cfrzr':  'Categorical freezing rain',
    'ci':     'Sea-ice cover',
    'cicep':  'Categorical ice pellets',
    'cin':    'Convective inhibition',
    'cnwat':  'Plant canopy surface water',
    'crain':  'Categorical rain',
    'csnow':  'Categorical snow',
    'dlwrf':  'Downward long-wave radiation flux',
    'dswrf':  'Downward short-wave radiation flux',
    'fricv':  'Frictional velocity',
    'gh':     'Geopotential Height',
    'gust':   'Wind speed (gust)',
    'hindex': 'Haines Index',
    'hlcy':   'Storm relative helicity',
    'hpbl':   'Planetary boundary layer height',
    'lftx':   'Surface lifted index',
    'lhtfl':  'Latent heat net flux',
    'lsm':    'Land-sea mask',
    'ltng':   'Lightning',
    'maxrh':  'Maximum relative humidity',
    'minrh':  'Minimum Relative Humidity',
    'mslet':  'MSLP (Eta model reduction)',
    'mstav':  'Moisture availability',
    'orog':   'Orography',
    'pli':    'Parcel lifted index (to 500 hPa)',
    'poros':  'Soil porosity',
    'pres':   'Pressure',
    'prmsl':  'Pressure reduced to MSL',
    'pwat':   'Precipitable water',
    'q':      'Specific humidity',
    'r':      'Relative humidity',
    'refc':   'Maximum/Composite radar reflectivity',
    'refd':   'Derived radar reflectivity',
    'rlyrs':  'Number of soil layers in root zone',
    'sde':    'Snow depth',
    'sdwe':   'Water equivalent of accumulated snow depth',
    'shtfl':  'Sensible heat net flux',
    'slt':    'Soil type',
    'smdry':  'Direct evaporation cease (soil moisture)',
    'smref':  'Transpiration stress-onset (soil moisture)',
    'snowc':  'Snow cover',
    'soill':  'Liquid volumetric soil moisture (non-frozen)',
    'soilw':  'Volumetric soil moisture content',
    'sp':     'Surface pressure',
    'sr':     'Surface roughness',
    'ssw':    'Soil moisture content',
    'st':     'Soil Temperature',
    't':      'Temperature',
    'tcc':    'Total Cloud Cover',
    'tke':    'Turbulent kinetic energy',
    'tmax':   'Maximum temperature',
    'tmin':   'Minimum temperature',
    'tp':     'Total Precipitation',
    'u':      'U component of wind',
    'ulwrf':  'Upward long-wave radiation flux',
    'uswrf':  'Upward short-wave radiation flux',
    'v':      'V component of wind',
    'veg':    'Vegetation',
    'vgtyp':  'Vegetation Type',
    'vis':    'Visibility',
    'VRATE':  'Ventilation Rate',
    'vucsh':  'Vertical u-component shear',
    'vvcsh':  'Vertical v-component shear',
    'w':      'Vertical velocity',
    'wilt':   'Wilting Point',
    'wz':     'Geometric vertical velocity',
}


def normalize_reftime(reftime=None):
    '''Normalize an arbitrary reference time to a valid one.

    Times may be strings, datetime objects, or `None` for the current reference
    time. Refrence times are converted to UTC and rounded to the previous 0h,
    6h, 12h or 18h mark. Strings take the format '%Y%m%dT%H%M'and are assumed
    to be UTC.

    Args:
        reftime (datetime or string):
            The reference time to prepare.
            Defaults to the most recent reference time.

    Returns (datetime):
        A valid reference time.
    '''
    # Default to most recent reference time
    if not reftime:
        reftime = datetime.now(timezone.utc)

    # Convert strings
    if isinstance(reftime, str):
        reftime = datetime.strptime(reftime, '%Y%m%dT%H%M')
        reftime = reftime.replace(tzinfo=timezone.utc)

    # Convert to UTC
    reftime = reftime.astimezone(timezone.utc)

    # Round to the previous 0h, 6h, 12h, or 18h
    hour = (reftime.hour // 6) * 6
    reftime = reftime.replace(hour=hour, minute=0, second=0, microsecond=0)

    return reftime


def native_reftime(ds, tz=None):
    '''Get the reftime of a dataset as a native datetime.

    This method extracts the first value along the reftime axis.
    By default, a NAM dataset only has one reftime.

    Example:
        Get the third value along the reftime dimension
        ```
        nam.native_reftime(ds.isel(reftime=2))
        ```

    Args:
        ds (xr.Dataset):
            A NAM dataset.
        tz (timezone):
            The data is converted to this timezone.
            The default is eastern standard time, the timezone of the data.
    '''
    if not tz:
        tz = timezone(timedelta(hours=-5), name='US/Eastern')

    reftime = (ds.reftime.data[0]
        .astype('datetime64[ms]')     # truncate from ns to ms (lossless for NAM data)
        .astype('O')                  # convert to native datetime
        .replace(tzinfo=timezone.utc) # mark the timezone as UTC
        .astimezone(tz))              # convert to given timezone

    return reftime


def latlon_index(lats, lons, pos):
    '''Find the index of the cell nearest to `pos`.

    Args:
        lats (2d array):
            The latitudes for each cell of the grid.
        lons (2d array):
            The longitudes for each cell of the grid.
        pos (float, float):
            The position as a `(latitude, longitude)` pair.

    Returns (int, int):
        The index of the cell nearest to `pos`.
    '''
    latlons = np.stack((lats.flatten(), lons.flatten()), axis=-1)
    target = np.array([pos])
    dist = sp.spatial.distance.cdist(target, latlons)
    am = np.argmin(dist)
    i, j = np.unravel_index(am, lats.shape)
    return i, j


def latlon_subset(lats, lons, center, apo):
    '''Build a slice to subset data based on lats and lons.

    Example:
        Subset projected data given the lats and lons:
        ```
        data, lats, lons = ...
        subset = latlon_subset(lats, lons)
        data[subset], lats[subset], lons[subset]
        ```

    Args:
        lats (2d array):
            The latitudes for each cell of the grid.
        lons (2d array):
            The longitudes for each cell of the grid.
        center (float, float):
            The center of the subset as a `(lat, lon)` pair.
        apo (int):
            The apothem of the subset in grid units,
            i.e. the number of cells from the center to the edge.

    Returns (slice, slice):
        A pair of slices characterizing the subset.
    '''
    i, j = latlon_index(lats, lons, center)
    return slice(i - apo, i + apo + 1), slice(j - apo, j + apo + 1)


def proj_coords(lats, lons):
    '''Transform geographic coordinates into the NAM218 projection.

    The input is a geographic area described by a pair of 2D arrays giving the
    latitude and longitude of each cell. The input must map to a square area in
    the projected coordinates. The values returned are 1D indices along both
    the x and y axes.

    The output is undefined if the input does not map to a square area.

    Args:
        lats (2D floats): The latitude at each cell.
        lons (2D floats): The longitude at each cell.

    Returns:
        y (1D floats): The coordinates for the y axis in meters.
        x (1D floats): The coordinates for the x axis in meters.
    '''
    unproj = ccrs.PlateCarree()
    coords = NAM218_PROJ.transform_points(unproj, lons, lats)
    x, y = coords[:,:,0], coords[:,:,1]
    x, y = x[0,:], y[:,0]
    x, y = np.copy(x), np.copy(y)
    return y, x


def plot_geo(data, show=True, block=False):
    '''A helper to plot geographic data.

    Args:
        data (xr.DataArray):
            The data to plot.
        show (bool):
            If True, show the plot immediately.
        block (bool):
            The blocking behavior when showing the plot.

    Example:
        Plot the 0-hour forecast of surface temperature:
        ```
        ds = load_nam()
        plot_geo(ds['t_surface'].isel(reftime=0, forecast=0))
        ```
    '''
    state_boarders = cf.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', facecolor='none')
    ax = plt.subplot(projection=NAM218_PROJ)
    ax.add_feature(state_boarders, edgecolor='black')
    ax.add_feature(cf.COASTLINE)
    data.plot(ax=ax)
    if show:
        plt.show(block=block)


def read_grib(path, features=FEATURE_SUBSET, geo=GEO_SUBSET):
    '''Processes a GRIB file into a list of `xr.Variable`s.

    Args:
        path (str or Path):
            The path to the GRIB file.
        features (list of str):
            The features to extract.
        geo (pair of slice):
            The subset of the grid to extract.

    Returns (list of xr.Variable):
        A list of variables extracted from the GRIB file.
    '''
    logger.info(f'processing {path}')

    # Open grib and select the variable subset.
    try:
        grbs = pygrib.open(str(path))
        grbs = grbs.select(shortName=features)
    except Exception as e:
        # read_grib gets called in parallel, so when it dies
        # we need to log somthing to identify which one died
        logger.warning(f'failed to read {path}: {e}')
        raise e

    # Infer the forecast hour.
    # - First check within the GRIB itself. All features in the file are of
    #   the same forecast hour, so we pull from the first feature.
    # - Otherwise check the path. The NAMLoader will include the forecast hour
    #   in the file name using the same format as the PROD_URL.
    match = re.search('t([0-9][0-9])z', str(path))
    if 'forecastTime' in grbs[1].keys():
        forecast = grbs[1].forecastTime
    elif match:
        forecast = int(match[1])
    else:
        raise RuntimeError('Cannot determine forecast hour')

    # Convert the forecast hour into a proper timedelta.
    forecast = np.timedelta64(forecast, 'h')

    # Collect the relevant features into a list of `xr.Variable`.
    variables = []
    for g in grbs:

        # Try to get a good layer name
        layer_type = g.typeOfLevel
        if layer_type == 'unknown':
            try:
                layer_type = 'z' + str(g.typeOfFirstFixedSurface)
            except:
                layer_type = 'z' + str(g.indicatorOfTypeOfLevel)

        # Try to make a good variable name
        name = '_'.join([g.shortName, layer_type])

        # Get the official reference time
        reftime = datetime(g.year, g.month, g.day, g.hour, g.minute, g.second)
        reftime = np.datetime64(reftime)

        # Get the units of the layer
        try:
            layer_units = g.unitsOfFirstFixedSurface
        except:
            layer_units = 'unknown'

        # Get the value along the z-axis
        level = g.level

        # Get the lats, lons
        # and x, y coordinates
        lats, lons = g.latlons()                   # lats and lons are in (y, x) order
        lats, lons = lats[geo], lons[geo]          # subset geographic region
        lats, lons = np.copy(lats), np.copy(lons)  # release reference to the grib
        y, x = proj_coords(lats, lons)             # convert to projected coordinates

        # Get the data values
        values = g.values                   # values are in (y, x) order
        values = values[geo]                # subset geographic region
        values = np.copy(values)            # release reference to the grib
        values = np.expand_dims(values, 0)  # (z, y, x)
        values = np.expand_dims(values, 0)  # (forecast, z, y, x)
        values = np.expand_dims(values, 0)  # (reftime, forecast, z, y, x)

        # The metadata associated with this variable.
        attrs = {
            'standard_name': g.cfName or g.name.replace(' ', '_').lower(),
            'short_name': g.shortName,
            'layer_type': layer_type,
            'units': g.units,
            'grid_mapping': 'NAM218',
        }

        # The names of the data axes.
        dims = ['reftime', 'forecast', layer_type, 'y', 'x']

        # The coordinates of the data.
        #
        # There are two types of coordinates:
        # - Dimension coordinates are the lables for the axes of the data.
        #   There are 5 dimensions: `reftime`, `forecast`, `z`, `y`, and `x`.
        # - Non-dimension coordinates do not label specific axes but may be
        #   aligned to other coordinates. The `lat` and `lon` coordinates are
        #   aligned to the `y` and `x` coordinates, and the `NAM218` coordinate
        #   describes the mapping from `lat` and `lon` to `x` and `y`.
        #
        # Note that the 'units' and 'calendar' attributes are automagically
        # attached to datetime and timedelta coordinates, i.e. reftime and forecase.
        #
        # http://xarray.pydata.org/en/stable/data-structures.html#coordinates
        # http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html
        coords = {
            'NAM218': ([], 0, NAM218_PROJ_CF),
            'lat': (('y', 'x'), lats, {
                'standard_name': 'latitude',
                'long_name': 'latitude',
                'units': 'degrees_north',
            }),
            'lon': (('y', 'x'), lons, {
                'standard_name': 'longitude',
                'long_name': 'longitude',
                'units': 'degrees_east',
            }),
            'reftime': ('reftime', [reftime], {
                'standard_name': 'forecast_reference_time',
                'long_name': 'reference time',
            }),
            'forecast': ('forecast', [forecast], {
                'standard_name': 'forecast_period',
                'long_name': 'forecast period',
                'axis': 'T',
            }),
            layer_type: (layer_type, [level], {
                'units': layer_units,
                'axis': 'Z',
            }),
            'y': ('y', y, {
                'standard_name': 'projection_y_coordinate',
                'units': 'm',
                'axis': 'Y',
            }),
            'x': ('x', x, {
                'standard_name': 'projection_x_coordinate',
                'units': 'm',
                'axis': 'X',
            }),
        }

        arr = xr.DataArray(name=name, data=values, dims=dims, coords=coords, attrs=attrs)
        variables.append(arr)

    return variables


def read_gribs(paths, features=FEATURE_SUBSET, geo=GEO_SUBSET):
    '''Process a set of GRIBs into an `xr.Dataset`.

    The algorithm works as follows:
    1. First we extract the variables from each file.
    2. Then we sort the variables into a tree using the complex key:
           (var_name, z_value, forecast_hour, reference_time)
    3. From the bottom up, we concatenate siblings, except at the top level.
    4. The top level becomes the set of the reconstructed variables,
       so we use `xr.merge` to create the dataset from these.

    Args:
        paths (strs or Paths):
            Paths to the GRIB file.
        features (list of str):
            The features to extract.
        geo (pair of slice):
            The subset of the grid to extract.

    Returns (xr.Dataset):
        A single dataset combining all of these GRIBS.
    '''
    def key(v):
        layer_type = v.attrs['layer_type']
        var_name = v.name
        z = v[layer_type].data[0]
        forecast = v['forecast'].data[0]
        reftime = v['reftime'].data[0]
        return (var_name, z, forecast, reftime)

    def axis_name(key):
        var_name = key[0]
        var_type, layer_type = var_name.split('_', 1)
        return layer_type

    with ThreadPoolExecutor() as pool:
        variables = pool.map(read_grib, paths, repeat(features), repeat(geo))
        variables = (v for vs in variables for v in vs) # flatten the nested structure
        variables = sorted(variables, key=key)
        variables = (xr.concat(g, dim='reftime')    for _, g in groupby(variables, lambda v: key(v)[:3]))
        variables = (xr.concat(g, dim='forecast')   for _, g in groupby(variables, lambda v: key(v)[:2]))
        variables = (xr.concat(g, dim=axis_name(k)) for k, g in groupby(variables, lambda v: key(v)[:1]))
        return xr.merge(variables, join='inner')


def clean(ds):
    '''Clean up some oddities that varry between forecasts.
    '''
    reftime = ds['reftime'].values[0]

    # Only use the first 36 hours of forecast (plus the 0 hour).
    # Some datasets contain the full 84 hours of forecast.
    ds = ds.isel(forecast=slice(37))

    # # Drop spatial coordinates.
    # # Due to rounding error, some datasets might not align exactly
    # # along these coordinates, causing xarray to freak out.
    ds = ds.drop('lat')
    ds = ds.drop('lon')
    ds = ds.drop('x')
    ds = ds.drop('y')

    # Drop cloud-top visibility
    # It is inconsistently avaliable.
    if 'cloudTop' in ds:
        ds = ds.drop('cloudTop')
        ds = ds.drop('vis_cloudTop')

    # Rename the 'z200' dimension to 'entireAtmosphere'.
    # After 2017-04-01, the grib files report this layer as 'unknown',
    # so the name is generated from the numeric layer indicator, 200.
    # TODO: Perhaps I can cleanup the old data and preprocessing code
    if 'z200' in ds:
        ds = ds.rename({'z200': 'entireAtmosphere'})
        ds = ds.rename({'tcc_z200': 'tcc_entireAtmosphere'})

    return ds


class NAMLoader:
    '''A class to download, subsets, and cache NAM forecasts.

    A `NAMLoader` downloads NAM-NMM forecasts from NOAA, subsets their features
    and geographic scope, converts the data to netCDF, and caches the result.

    The feature and geographic subsets are global constants `FEATURE_SUBSET`
    and `GEO_SUBSET`. Do NOT alter these. Caching forecasts with a different
    feature or geographic subset is NOT a good idea.

    TODO:
        - describe the data format
    '''
    class CacheMiss(Exception):
        pass

    def __init__(self,
            cache_dir='./NAM-NMM',
            force_download=False,
            fail_fast=False,
            save_netcdf=True,
            save_gribs=False):
        '''Creates a loader for NAM data.

        Args:
            cache_dir (Path or str):
                The path to the cache.
            force_download (bool):
                Force the download and processing of grib files,
                ignoring any cached grib or netCDF datasets.
            fail_fast (bool):
                If true, the download errors are treated as fatal.
                Otherwise downloads are retried with exponential backoff.
                This overrides the `max_tries` argument of the `download`
                method.
            save_netcdf (bool):
                If true, cache the dataset as a netCDF file.
            save_gribs (bool):
                If true, cache the dataset as a series of GRIB files.
        '''
        self.cache_dir = Path(cache_dir)
        self.save_netcdf = save_netcdf
        self.save_gribs = save_gribs
        self.force_download = force_download
        self.fail_fast = fail_fast

    def remote_gribs(self, reftime=None):
        '''An iterator over the URLs for a forecast.

        The production URLs from NCEP are used if `reftime` is within the past
        week. Otherwise the archive URLs provided by NCDC are used.

        Note that the archive only goes back 11 months.

        Args:
            reftime (datetime):
                The reference time.

        Yields (Path):
            URLs to GRIB files released by NOAA.
        '''
        reftime = normalize_reftime(reftime)
        now = datetime.now(timezone.utc)
        delta = now - reftime
        if delta.days > 7:
            if reftime < datetime(year=2017, month=4, day=1, tzinfo=timezone.utc):
                url_fmt = ARCHIVE_URL_1
            else:
                url_fmt = ARCHIVE_URL_2
        else:
            url_fmt = PROD_URL
        for i in FORECAST_PERIOD:
            yield url_fmt.format(ref=reftime, forecast=i)

    def local_gribs(self, reftime=None):
        '''An iterator over the local GRIB paths for a forecast.

        The files may not exists.

        Args:
            reftime (datetime):
                The reference time.

        Yields (Path):
            Paths to GRIB files under the `cache_dir`.
        '''
        reftime = normalize_reftime(reftime)
        prefix_fmt = 'nam.{ref.year:04d}{ref.month:02d}{ref.day:02d}'
        filename_fmt = 'nam.t{ref.hour:02d}z.awphys{forecast:02d}.tm00.grib'
        for i in FORECAST_PERIOD:
            prefix = prefix_fmt.format(forecast=i, ref=reftime)
            filename = filename_fmt.format(forecast=i, ref=reftime)
            yield self.cache_dir / prefix / filename

    def local_cdf(self, reftime=None):
        '''The path to a local netCDF file for a forecast.

        The file may not exist.

        Args:
            reftime (datetime):
                The reference time to load.

        Returns (Path):
            A path to a netCDF file under the `cache_dir`.
        '''
        reftime = normalize_reftime(reftime)
        prefix = f'nam.{reftime.year:04d}{reftime.month:02d}{reftime.day:02d}'
        filename = f'nam.t{reftime.hour:02d}z.awphys.tm00.nc'
        return self.cache_dir / prefix / filename

    def download(self, reftime=None, max_tries=8, timeout=10):
        '''Download the GRIB files for this dataset.

        Args:
            reftime (datetime):
                The reference time to download.
            max_tries (int):
                The maximum number of failed downloads for a single file
                before raising an `IOError`. This option is ignored if
                `fail_fast` is set on the NAMLoader.
            timeout (int):
                The network timeout in seconds.
                Note that the government servers can be kinda slow.
        '''
        reftime = normalize_reftime(reftime)

        # fail_fast overrides max_tries
        if self.fail_fast: max_tries = 1

        local_gribs = self.local_gribs(reftime)
        remote_gribs = self.remote_gribs(reftime)
        for path, url in zip(local_gribs, remote_gribs):
            # Ensure that we have a destination to download into.
            path.parent.mkdir(exist_ok=True)

            # Skip existing files unless `force_download`.
            if not self.force_download and path.exists():
                continue

            for i in range(max_tries):
                # Perform a streaming download because the files are big.
                try:
                    with path.open('wb') as fd:
                        logger.info(f'downloading {url}')
                        r = requests.get(url, timeout=timeout, stream=True)
                        r.raise_for_status()
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)
                    break

                # IOError includes both system and HTTP errors.
                # Retry with exponential backoff.
                except IOError as err:
                    logger.warning(err)
                    path.unlink()
                    if i + 1 == max_tries:
                        logger.error('download failed, giving up')
                        raise err
                    else:
                        delay = 2**i
                        logger.warning(f'download failed, retrying in {delay}s')
                        sleep(delay)
                        continue

                # Delete partial file in case of keyboard interupt etc.
                except (Exception, SystemExit, KeyboardInterrupt) as err:
                    path.unlink()
                    raise err

    def load_from_cache(self, reftime=None):
        '''Load the forecast for this reference time from the cache.

        Args:
            reftime (datetime):
                The reference time to load.

        Returns:
            An `xr.Dataset` describing this forecast.
        '''
        reftime = normalize_reftime(reftime)
        path = self.local_cdf(reftime)

        if path.exists():
            logger.debug(f'loading forecast {reftime} from cache')
            ds = xr.open_dataset(str(path), autoclose=True, chunks={})
            return ds
        else:
            raise NAMLoader.CacheMiss(reftime)

    def load_from_grib(self, reftime=None):
        '''Load the dataset for this forecast from the GRIB files.

        If the files are not cached, they will be downloaded.

        Args:
            reftime (datetime):
                The reference time to load.

        Returns:
            An `xr.Dataset` describing this forecast.
        '''
        reftime = normalize_reftime(reftime)
        logger.debug(f'loading forecast {reftime} from gribs')

        self.download(reftime) # ensure the files exist.
        dataset = read_gribs(self.local_gribs(reftime))

        if self.save_netcdf:
            cdf_path = self.local_cdf(reftime)
            logger.info(f'writing {cdf_path}')
            dataset.to_netcdf(str(cdf_path))

        if not self.save_gribs:
            logger.info('deleting local gribs')
            for grib_path in self.local_gribs(reftime):
                grib_path.unlink()

        return dataset

    def load_one(self, reftime=None):
        '''Load a forecast for some reference time,
        downloading and preprocessing GRIBs as necessary.

        If the dataset exists as a local netCDF file, it is loaded and
        returned. Otherwise, any missing GRIB files are downloaded and
        preprocessed into an xarray Dataset. The dataset is then saved as a
        netCDF file, the GRIBs are deleted, and the dataset is returned.

        Args:
            reftime (datetime):
                The reference time to load. It is rounded down to the
                previous 6 hour mark. It may be given as a string in the
                format '%Y%m%dT%H%M'. The default is the current time.

        Returns (xr.Dataset):
            A dataset for the forecast at the given reference time.
        '''
        reftime = normalize_reftime(reftime)

        logger.debug(f'loading forecast at {reftime}')

        # `force_download` means we must load from grib files.
        if self.force_download:
            return self.load_from_grib(reftime)

        # Otherwise, try loading from the cache first.
        try:
            return self.load_from_cache(reftime)

        except NAMLoader.CacheMiss as e:
            logger.debug(f'cache miss: {e}')
            return self.load_from_grib(reftime)

    def load(self, *reftimes):
        '''Load and combine forecasts for some reference times,
        downloading and preprocessing GRIBs as necessary.

        If the dataset exists as a local netCDF file, it is loaded and
        returned. Otherwise, any missing GRIB files are downloaded and
        preprocessed into an xarray Dataset. The dataset is then saved as a
        netCDF file, the GRIBs are deleted, and the dataset is returned.

        Args:
            reftimes (datetime):
                The reference times to load. They are rounded down to the
                previous 6 hour mark. They may be given as a string in the
                format '%Y%m%dT%H%M'. The default is to load only the most
                recent forecast.

        Returns (xr.Dataset):
            Returns a single dataset containing all forecasts at the given
            reference times. Some data may be dropped when combining forecasts.
        '''
        if not reftimes:
            return self.load_one()
        else:
            datasets = (self.load_one(r) for r in reftimes)
            datasets = (clean(ds) for ds in datasets)
            return xr.concat(datasets, dim='reftime')

    def load_range(self, start='20160901T0000', stop=None):
        '''Load  and combine forecasts for a range of reference times.

        NOTE: This method only loads data from the cache.

        Args:
            start (datetime):
                The first time in the range.
                The default is 2016-09-01 00:00
            stop (datetime):
                The last time in the range.
                The default is the current time.

        Returns (xr.Dataset):
            Returns a single dataset containing all forecasts at the given
            reference times. Some data may be dropped when combining forecasts.
        '''
        start = normalize_reftime(start)
        stop = normalize_reftime(stop)
        logger.info(f'loading forecasts from {start} to {stop}')

        datasets = []
        delta = timedelta(hours=6)
        while start <= stop:
            try:
                ds = self.load_from_cache(start)
                ds = clean(ds)
                datasets.append(ds)
            except NAMLoader.CacheMiss:
                pass
            start += delta

        logger.info('joining forecasts')
        return xr.concat(datasets, dim='reftime')


def load_range(start='20160901T0000', stop=None, **kwargs):
    '''Load  and combine forecasts for a range of reference times.

    NOTE: This method only loads data from the cache.

    Args:
        start (datetime):
            The first time in the range.
            The default is 2016-09-01 00:00
        stop (datetime):
            The last time in the range.
            The default is the current time.

    Returns (xr.Dataset):
        Returns a single dataset containing all forecasts at the given
        reference times. Some data may be dropped when combining forecasts.
    '''
    loader = NAMLoader(**kwargs)
    return loader.load_range(start, stop)


def load_nam(*reftimes, **kwargs):
    '''Load and combine forecasts for some reference times,
    downloading and preprocessing GRIBs as necessary.

    If the dataset exists as a local netCDF file, it is loaded and
    returned. Otherwise, any missing GRIB files are downloaded and
    preprocessed into an xarray Dataset. The dataset is then saved as a
    netCDF file, the GRIBs are deleted, and the dataset is returned.

    Args:
        reftimes (datetime):
            The reference times to load. They are rounded down to the
            previous 6 hour mark. They may be given as a string in the
            format '%Y%m%dT%H%M'. The default is to load only the most
            recent forecast.

    Returns (xr.Dataset):
        Returns a single dataset containing all forecasts at the given
        reference times. Some data may be dropped when combining forecasts.
    '''
    loader = NAMLoader(**kwargs)
    return loader.load(*reftimes)
