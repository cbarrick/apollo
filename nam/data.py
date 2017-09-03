#!/usr/bin/env python3
'''A NAM dataset loader.

> The North American Mesoscale Forecast System (NAM) is one of the
> major weather models run by the National Centers for Environmental
> Prediction (NCEP) for producing weather forecasts. Dozens of weather
> parameters are available from the NAM grids, from temperature and
> precipitation to lightning and turbulent kinetic energy.

> As of June 20, 2006, the NAM model has been running with a non-
> hydrostatic version of the Weather Research and Forecasting (WRF)
> model at its core. This version of the NAM is also known as the NAM
> Non-hydrostatic Mesoscale Model (NAM-NMM).

Most users will be interested in the `load` function which loads
the data for a single NAM-NMM run at a particular reference time,
downloading and preprocessing GRIB files if needed. The actual data
loading logic is encapsulated in the `NAMLoader` class which can be
used for finer grain control over the preprocessing and file system
usage or to load different NAM datasets like NAM-ANL.

The data loading logic works like this:

1. If a netCDF file exists for the dataset, it is loaded immediately
   without any preprocessing.
2. Otherwise any GRIB files required for building the dataset are
   downloaded if they do not already exist.
3. The data is then extracted from the GRIBs. The raw data is subsetted
   to an area encompasing Georgia, and only a subset of the features
   are extracted. The level and time axes are reconstructed from
   multiple GRIB features.
4. The dataset is then saved to a netCDF file, and the GRIB files are
   removed.

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
'''

from datetime import datetime, timedelta, timezone
from itertools import groupby
from logging import getLogger
from pathlib import Path
from time import sleep

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial.distance
import pygrib
import requests
import xarray as xr


# Module level logger
logger = getLogger(__name__)


# URLs of remote grib files.
# PROD_URL typically has the most recent 7 days.
# ARCHIVE_URL typically has the most recent 11 months, about 1 week behind.
PROD_URL = 'http://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.{ref.year:04d}{ref.month:02d}{ref.day:02d}/nam.t{ref.hour:02d}z.awphys{forecast:02d}.tm00.grib2'
ARCHIVE_URL_1 = 'https://nomads.ncdc.noaa.gov/data/meso-eta-hi/{ref.year:04d}{ref.month:02d}/{ref.year:04d}{ref.month:02d}{ref.day:02d}/nam_218_{ref.year:04d}{ref.month:02d}{ref.day:02d}_{ref.hour:02d}00_{forecast:03d}.grb'
ARCHIVE_URL_2 = 'https://nomads.ncdc.noaa.gov/data/meso-eta-hi/{ref.year:04d}{ref.month:02d}/{ref.year:04d}{ref.month:02d}{ref.day:02d}/nam_218_{ref.year:04d}{ref.month:02d}{ref.day:02d}_{ref.hour:02d}00_{forecast:03d}.grb2'


# The standard forecast period of the NAM-NMM dataset.
FORECAST_PERIOD = tuple(range(0, 36)) + tuple(range(36, 85, 3))


# The projection of the NAM-NMM dataset as a `cartopy.crs.CRS`.
PROJ = ccrs.LambertConformal(
    central_latitude=25,
    central_longitude=265,
    standard_parallels=(25, 25),

    # The default cartopy globe is WGS 84, but
    # NAM assumes a spherical globe with radius 6,371.229 km
    globe=ccrs.Globe(ellipse=None, semimajor_axis=6371229, semiminor_axis=6371229),
)


# The projection of the NAM-NMM dataset as a CF convention grid mapping.
# Stored in the netCDF files when they are converted.
CF_PROJ = {
    'grid_mapping_name': 'lambert_conformal_conic',
    'latitude_of_projection_origin': 25.0,
    'longitude_of_central_meridian': 265.0,
    'standard_parallel': 25.0,
    'earth_radius': 6371229.0,
}


# The default features to load.
DEFAULT_FEATURES = ['dlwrf', 'dswrf', 'pres', 'vis', 'tcc', 't', 'r', 'u', 'v', 'w']


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


class NAMLoader:
    '''A downloader and preprocessor for NAM forecast data.

    NAM releases are downloaded from the web and the variables and geographic
    region are subsetted. The result can be saved back to disk as a netCDF
    file.

    TODO:
        - describe the data format
    '''

    def __init__(self,
            ref_time=None,
            features=DEFAULT_FEATURES,
            center=(32.8, -83.6),
            apo=50,
            forecast_period=FORECAST_PERIOD,
            data_dir='.',
            url_fmt=None,
            local_grib_fmt='nam.{ref.year:04d}{ref.month:02d}{ref.day:02d}/nam.t{ref.hour:02d}z.awphys{forecast:02d}.tm00.grib',
            local_cdf_fmt='nam.{ref.year:04d}{ref.month:02d}{ref.day:02d}/nam.t{ref.hour:02d}z.awphys.tm00.nc',
            save_netcdf=True,
            keep_gribs=False,
            force_download=False,
            fail_fast=False):
        '''Creates a loader for NAM data.

        Args:
            ref_time (datetime):
                The default reference time of the data set. It is rounded down
                to the previous model run. It may be given as a string with the
                format '%Y%m%d %H%M'. The default is the most recent release.
            forecast_period (list of ints):
                The forecast hours to load.
            features (list of str):
                Filter the dataset to only include these features.
                This argument is ignored when loading from netCDF.
            center (float, float):
                The geographic center of the of the data as a `(lat, lon)` pair.
            apo (int):
                The apothem of the geographic subset in grid units,
                i.e. the number of cells from the center to the edge.
            data_dir (Path):
                The base path for the dataset.
            url_fmt (string):
                The URL format to download missing data.
                The default is automatically selected from the reference time.
            local_grib_fmt (string):
                The name format for local grib files.
            local_cdf_fmt (string):
                The name format for local netCDF files.
            save_netcdf (bool):
                If true, save the dataset to a netCDF file.
                This argument defines the behavior of the `load` method,
                and is ignored when calling `load_from_grib` directly.
            keep_gribs (bool):
                If true, the GRIB files are not deleted.
                This argument defines the behavior of the `load` method,
                and is ignored when calling `load_from_grib` directly.
            force_download (bool):
                Force the download and processing of grib files,
                ignoring any existing grib or netCDF datasets.
                This argument defines the behavior of the `load` method,
                and is ignored when calling `download` or `load_from_grib`
                directly.
            fail_fast (bool):
                If true, the download errors are treated as fatal.
                Otherwise downloads are retried with exponential backoff.
                This argument defines the behavior of the `load` method,
                and is ignored when calling `download` or `load_from_grib`
                directly.
        '''
        self.ref_time = self.prepare_ref_time(ref_time or datetime.now())
        self.features = features
        self.center = center
        self.apo = apo
        self.forecast_period = tuple(forecast_period)
        self.data_dir = Path(data_dir)
        self.url_fmt = url_fmt or self.automatic_url_fmt(self.ref_time)
        self.local_grib_fmt = local_grib_fmt
        self.local_cdf_fmt = local_cdf_fmt
        self.save_netcdf = save_netcdf
        self.keep_gribs = keep_gribs
        self.force_download = force_download
        self.fail_fast = fail_fast

        self._data = None

    def load(self):
        '''Load the dataset, downloading and preprocessing GRIBs as necessary.

        If the dataset exists as a local netCDF file, it is loaded and
        returned. Otherwise, any missing GRIB files are downloaded and
        preprocessed into an xarray Dataset. The dataset is then saved as a
        netCDF file, the GRIBs are deleted, and the dataset is returned.
        '''
        logger.info('loading dataset for {}'.format(self.ref_time))

        if not self.force_download and self.local_cdf.exists():
            return self.load_from_cdf()

        else:
            return self.load_from_grib(
                save_netcdf=self.save_netcdf,
                keep_gribs=self.keep_gribs,
                force_download=self.force_download,
                fail_fast=self.fail_fast)

    def load_from_cdf(self):
        '''Load the dataset from the netCDF file for this release.

        Returns:
            An `xr.Dataset` describing this release.
        '''
        # TODO: verify and extract only the requested feature/geo subsets.
        logger.info('loading netCDF data')
        data = xr.open_dataset(str(self.local_cdf))
        return data

    def load_from_grib(self,
            save_netcdf=False,
            keep_gribs=True,
            force_download=False,
            fail_fast=True):
        '''Load the dataset from the GRIB files for this release.

        If the files do not exist locally, they will be downloaded.

        Args:
            save_netcdf (bool):
                If true, save the dataset to a netCDF file.
            keep_gribs (bool):
                If true, the GRIB files are not deleted.
            force_download (bool):
                Force the download and processing of grib files,
                ignoring any existing grib or netCDF datasets.
            fail_fast (bool):
                If true, the download errors are treated as fatal.
                Otherwise downloads are retried with exponential backoff.

        Returns:
            An `xr.Dataset` describing this release.
        '''
        logger.info('loading grib data')

        # Ensure the files exist.
        self.download(force_download, fail_fast)

        # Process the individual grib files into a list of `xr.Variable`s.
        variables = []
        for path in self.local_gribs:
            vs = self.process_grib(path)
            variables.extend(vs)

        # To combine the variables into an `xr.DataSet`,
        # they must be sorted by this key.
        def key(v):
            name = v.name
            layer_type = v.attrs['layer_type']
            forecast = v.forecast.data[0]
            z = v[layer_type].data[0]
            return (name, layer_type, z, forecast)

        logger.info('Sorting variables')
        variables = sorted(variables, key=key)

        logger.info('Reconstructing the forecast dimension')
        tmp = []
        for k, g in groupby(variables, lambda v: key(v)[:3]):
            v = xr.concat(g, dim='forecast')
            tmp.append(v)
        variables = tmp

        logger.info('Reconstructing the z dimensions')
        tmp = []
        for k, g in groupby(variables, lambda v: key(v)[:2]):
            dim = k[1]
            v = xr.concat(g, dim=dim)
            tmp.append(v)
        variables = tmp

        logger.info('Collecting the dataset')
        dataset = xr.merge(variables, join='inner')

        if save_netcdf:
            logger.info('Saving as {}'.format(self.local_cdf))
            dataset.to_netcdf(str(self.local_cdf))

        if not keep_gribs:
            logger.info('Deleting local gribs')
            for grib_path in self.local_gribs:
                grib_path.unlink()

        return dataset

    def process_grib(self, path):
        '''Reads and processes a grib file at the given path.

        Returns (list of xr.Variable):
            A list of variables extracted from the grib file.
        '''
        logger.info('Processing {}'.format(path))

        # Open grib and select the variable subset.
        grbs = pygrib.open(str(path))
        grbs = grbs.select(shortName=self.features)

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
            ref_time = datetime(g.year, g.month, g.day, g.hour, g.minute, g.second)
            ref_time = np.datetime64(ref_time)

            # Get the forecast time as an offset from the reference time.
            try:
                forecast = np.timedelta64(g.forecastTime, 'h')
            except:
                forecast = np.timedelta64(g.dataTime, 'h')

            # Get the units of the layer
            try:
                layer_units = g.unitsOfFirstFixedSurface
            except:
                layer_units = 'unknown'

            # Get the value of the z-axis
            level = g.level

            # Get the lats, lons
            # and x, y coordinates
            lats, lons = g.latlons()                     # lats and lons are in (y, x) order
            lats, lons = lats[self.geo], lons[self.geo]  # subset geographic region
            lats, lons = np.copy(lats), np.copy(lons)    # release reference to the grib
            y, x = proj_coords(lats, lons)               # convert to projected coordinates

            # Get the data values
            values = g.values                   # values are in (y, x) order
            values = values[self.geo]           # subset geographic region
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

            # The data and metadata describing the coordinates.
            coords = {
                'NAM218': ([], 0, CF_PROJ),
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
                'reftime': ('reftime', [ref_time], {
                    'standard_name': 'forecast_reference_time',
                    'long_name': 'reference time',
                    # # units and calendar are handled automatically by xarray
                    # 'units': 'seconds since 1970-01-01 0:0:0',
                    # 'calendar': 'standard',
                }),
                'forecast': ('forecast', [forecast], {
                    'standard_name': 'forecast_period',
                    'long_name': 'forecast period',
                    'axis': 'T',
                    # # units and calendar are handled automatically by xarray
                    # 'units': 'seconds',
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

    def download(self, force_download=False, fail_fast=True, max_tries=8, timeout=10):
        '''Download the GRIB files for this dataset.

        Args:
            force_download (bool):
                Download grib files even if they already exist.
            fail_fast (bool):
                Raise an `IOError` after the first failed download.
                Overrides the `max_tries` argument.
            max_tries (int):
                The maximum number of failed downloads for a single file
                before raising an `IOError`.
            timeout (int):
                The network timeout in seconds.
                Note that the government servers can be kinda slow.
        '''
        if fail_fast:
            max_tries = 1

        for path, url in zip(self.local_gribs, self.remote_gribs):
            # Ensure that we have a destination to download into.
            path.parent.mkdir(exist_ok=True)

            # Skip existing files unless `force_download`.
            if not force_download and path.exists():
                continue

            for i in range(max_tries):
                # Perform a streaming download because the files are big.
                try:
                    with path.open('wb') as fd:
                        logger.info('Downloading {}'.format(url))
                        r = requests.get(url, timeout=timeout, stream=True)
                        r.raise_for_status()
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)
                    break

                # IOError includes both system and HTTP errors.
                # Retry with exponential backoff unless `fail_fast`.
                except IOError as err:
                    logger.warn(err)
                    path.unlink()
                    if i+1 == max_tries:
                        logger.error('Download failed, giving up')
                        raise err
                    else:
                        delay = 2**i
                        logger.warn('Download failed, retrying in {}s'.format(delay))
                        sleep(delay)
                        continue

                # Delete partial file in case of keyboard interupt etc.
                except (Exception, SystemExit, KeyboardInterrupt) as err:
                    path.unlink()
                    raise err

    def prepare_ref_time(self, ref_time):
        '''Convert an arbitrary reference time to a valid one.

        Args:
            ref_time (datetime or string):
                The reference time to prepare.

        Returns (datetime):
            A valid reference time.
        '''
        # Convert strings
        if isinstance(ref_time, str):
            ref_time = datetime.strptime(ref_time, '%Y%m%d %H%M')
            ref_time = ref_time.replace(tzinfo=timezone.utc)

        # Convert to UTC
        ref_time = ref_time.astimezone(timezone.utc)

        # Round to the previous 0h, 6h, 12h, or 18h
        hour = (ref_time.hour // 6) * 6
        ref_time = ref_time.replace(hour=hour, minute=0, second=0, microsecond=0)

        return ref_time

    def automatic_url_fmt(self, ref_time):
        '''Derive the url for data at a given reference time.

        Note that the appropriate URL depends on the current time and therefore
        is not stable. Do not depend on this output for an extended period.

        Args:
            ref_time (datetime):
                A valid reference time. To convert an arbitrary datetime to a
                valid one, use `NAMLoader.prepare_ref_time`.

        Returns (str):
            Either `PROD_URL`, `ARCHIVE_URL_1`, or `ARCHIVE_URL_2`.
        '''
        now = datetime.now(timezone.utc)
        days_delta = (now - ref_time).days
        if days_delta > 7:
            if ref_time < datetime(year=2017, month=4, day=1, tzinfo=timezone.utc):
                url_fmt = ARCHIVE_URL_1
            else:
                url_fmt = ARCHIVE_URL_2
        else:
            url_fmt = PROD_URL
        return url_fmt

    @property
    def geo(self):
        '''The geographic subset to extract from the grib files.

        Note that the 0-hour GRIB file MUST exist locally.

        Returns (slice, slice):
            A pair of slices characterizing the subset.
        '''
        if not hasattr(self, '_geo'):
            paths = tuple(self.local_gribs)
            first_file = str(paths[0])
            grbs = pygrib.open(first_file)
            g = grbs[1]  # indices start at 1
            lats, lons = g.latlons()
            self._geo = latlon_subset(lats, lons, center=self.center, apo=self.apo)
        return self._geo

    @property
    def remote_gribs(self):
        '''An iterator over the URLs of GRIB files for some reference time and forecast period.'''
        for i in self.forecast_period:
            url = self.url_fmt.format(forecast=i, ref=self.ref_time)
            yield url

    @property
    def local_gribs(self):
        '''An iterator over paths to local GRIB files for some reference time and forecast period.'''
        for i in self.forecast_period:
            p = self.local_grib_fmt.format(forecast=i, ref=self.ref_time)
            yield self.data_dir / Path(p)

    @property
    def local_cdf(self):
        '''The path to a local netCDF file for some reference time.'''
        p = self.local_cdf_fmt.format(ref=self.ref_time)
        return self.data_dir / Path(p)


def load(*args, **kwargs):
    '''Load a NAM-NMM dataset for the given reference time.

    See `NAMLoader` for a description of accepted arguments.
    '''
    loader = NAMLoader(*args, **kwargs)
    return loader.load()


def reftime(ds, tz=None):
    '''Returns the first value along the reftime dimension as a native datetime.

    Example:
        Get the third value along the reftime dimension
        ```
        nam.reftime(ds.isel(reftime=2))
        ```

    Args:
        ds (xr.DataSet):
            A NAM dataset.
        tz (timezone):
            The data is converted to this timezone.
            The default is eastern standard time.
    '''
    if not tz:
        tz = timezone(timedelta(hours=-5), name='US/Eastern')

    reftime = (ds.reftime.data[0]
        .astype('datetime64[ms]')     # truncate from ns to ms (lossless for NAM data)
        .astype('O')                  # convert to native datetime
        .replace(tzinfo=timezone.utc) # set timezone
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
    '''Build a slice to subset projected data based on lats and lons.

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
    '''Transform a grid of geographic coordinates into projection coordinates
    for the x and y axes.

    Args:
        lats (matrix): The latitudes for every cell of the map.
        lons (matrix): The longitudes for tevery cell on the map.

    Returns:
        x (vector): The coordinates for the x axis in meters.
        y (vector): The coordinates for the y axis in meters.
    '''
    unproj = ccrs.PlateCarree()
    coords = PROJ.transform_points(unproj, lons, lats)
    x, y = coords[:,:,0], coords[:,:,1]
    x, y = x[0,:], y[:,0]
    x, y = np.copy(x), np.copy(y)
    return y, x


def show(data):
    '''A helper to plot NAM data.

    Example:
        Plot the 0-hour forecast of surface temperature:
        ```
        ds = nam.load()
        nam.show(ds.t_surface.isel(reftime=0, forecast=0))
        ```
    '''
    state_boarders = cf.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', '50m', facecolor='none')
    ax = plt.subplot(projection=PROJ)
    ax.add_feature(state_boarders, edgecolor='black')
    ax.add_feature(cf.COASTLINE)
    data.plot(ax=ax)
    plt.show(block=False)


if __name__ == '__main__':
    import argparse
    import logging

    now = datetime.now(timezone.utc)

    parser = argparse.ArgumentParser(description='Download and preprocess the NAM-NMM dataset.')
    parser.add_argument('--log', type=str, help='Set the log level')
    parser.add_argument('--stop', type=lambda x: datetime.strptime(x, '%Y-%m-%dT%H00'), help='The last reference time')
    parser.add_argument('--start', type=lambda x: datetime.strptime(x, '%Y-%m-%dT%H00'), help='The first reference time')
    parser.add_argument('--fail-fast', action='store_true', help='Do not retry downloads')
    parser.add_argument('--keep-gribs', action='store_true', help='Do not delete grib files')
    parser.add_argument('-n', type=int, help='The number of most recent releases to process.')
    parser.add_argument('-f', '--forecast', type=int, metavar='N', default=len(FORECAST_PERIOD)-1, help='Only process the first N forecasts')
    parser.add_argument('dir', nargs='?', type=str, help='Base directory for downloads')
    args = parser.parse_args()

    log_level = args.log or 'INFO'
    logging.basicConfig(level=log_level, format='[{asctime}] {levelname}: {message}', style='{')

    data_dir = args.dir or '.'

    stop = args.stop.replace(tzinfo=timezone.utc) if args.stop else now
    delta = timedelta(hours=6)

    if args.start:
        start = args.start.replace(tzinfo=timezone.utc)
    elif args.n:
        start = stop - args.n * delta
    else:
        start = stop

    fail_fast = args.fail_fast
    keep_gribs = args.keep_gribs
    forecast_period = FORECAST_PERIOD[:args.forecast+1]

    while start <= stop:
        try:
            load(start, data_dir=data_dir, fail_fast=fail_fast, keep_gribs=keep_gribs, forecast_period=forecast_period)
        except Exception as e:
            logger.error(e)
            logger.error('Could not load data from {}'.format(start))
        start += delta
