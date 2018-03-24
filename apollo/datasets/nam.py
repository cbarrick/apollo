'''Provides access to a subset of the NAM-NMM dataset.

> The North American Mesoscale Forecast System (NAM) is one of the
> major weather models run by the National Centers for Environmental
> Prediction (NCEP) for producing weather forecasts. Dozens of weather
> parameters are available from the NAM grids, from temperature and
> precipitation to lightning and turbulent kinetic energy.

> As of June 20, 2006, the NAM model has been running with a non-
> hydrostatic version of the Weather Research and Forecasting (WRF)
> model at its core. This version of the NAM is also known as the NAM
> Non-hydrostatic Mesoscale Model (NAM-NMM).

Most users will be interested in the `open` and `open_range`
functions which selects forecasts by reference time. The actual data
loading logic is encapsulated in the `NamLoader` class.

The dataset live remotely. A live feed is provided by NCEP and an 11
month archive is provided by NCDC (both are divisions of NOAA). This
module caches the data locally, allowing us to build larger archives.
The remote dataset is provided in GRIB format, while this module uses
the netCDF format in its internal cache.

This module provides access to only a subset of the NAM-NMM dataset.
The geographic region is reduced and centered around Georgia, and only
as subset of the variables are provided.

Queries return an instance of `xarray.Dataset` where each variable has
exactly five dimensions: reftime, forecast, z, y, and x. The z-axis for
each variable has a different name depending on the type of index
measuring the axis, e.g. `z_ISBL`.
'''

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep
import logging

import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.spatial
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


def open(*reftimes, **kwargs):
    '''Load and combine forecasts for some reference times,
    downloading and preprocessing GRIBs as necessary.

    If the dataset exists as a local netCDF file, it is loaded and
    returned. Otherwise, any missing GRIB files are downloaded and
    preprocessed into an xarray Dataset. The dataset is then saved as a
    netCDF file, the GRIBs are deleted, and the dataset is returned.

    Args:
        reftimes (datetime64):
            The reference times to open.

    Returns (xr.Dataset):
        Returns a single dataset containing all forecasts at the given
        reference times. Some data may be dropped when combining forecasts.
    '''
    loader = NamLoader(**kwargs)
    return loader.open(*reftimes)


def open_range(start='2017-01-01', stop='today', **kwargs):
    '''Load and combine forecasts for a range of reference times.

    NOTE: This method only loads data from the cache.

    Args:
        start (datetime64):
            The first time in the range.
            The default is 2017-01-01T00:00
        stop (datetime64):
            The last time in the range.
            The default is the start of the current day.

    Returns (xr.Dataset):
        Returns a single dataset containing all forecasts at the given
        reference times. Some data may be dropped when combining forecasts.
    '''
    loader = NamLoader(**kwargs)
    return loader.open_range(start, stop)


def open_gribs(reftime='now', **kwargs):
    '''Load the forecasts from GRIB, downlading if they do not exist.

    Args:
        reftime (datetime64):
            The reference time to open.

    Returns:
        An `xr.Dataset` describing this forecast.
    '''
    loader = NamLoader(**kwargs)
    return loader.open_gribs(reftime)


def open_nc(reftime='now', **kwargs):
    '''Load the forecasts from a netCDF in the cache.

    Args:
        reftime (datetime64):
            The reference time to open.

    Returns:
        An `xr.Dataset` describing this forecast.
    '''
    loader = NamLoader(**kwargs)
    return loader.open_nc(reftime)


def normalize_reftime(reftime='now'):
    return np.datetime64(reftime, '6h')


def find_nearest(data, *points, **kwargs):
    '''Find the indices of `data` nearest to `points`.

    Returns:
        The unraveled indices into `data` of the cells nearest to `points`.
    '''
    n = len(data)
    shape = data[0].shape
    data = np.require(data).reshape(n, -1).T
    points = np.require(points).reshape(-1, n)
    idx = sp.spatial.distance.cdist(points, data, **kwargs).argmin(axis=1)
    return tuple(np.unravel_index(i, shape) for i in idx)


def diagonal_slice(data, a, b, **kwargs):
    '''Build a diagonal slice between the points nearest to `a` and `b`.

    The slice selects a cube where `a` and `b` are opposite corners.

    Returns:
        A list of slices to select the cube where each slices the
        corresponding axis in `data`.
    '''
    a, b = find_nearest(data, a, b, **kwargs)
    return [slice(i, j) for i, j in zip(a,b)]


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
        x (1D floats): The coordinates for the x axis in meters.
        y (1D floats): The coordinates for the y axis in meters.
    '''
    unproj = ccrs.PlateCarree()
    coords = NAM218_PROJ.transform_points(unproj, lons, lats)
    x, y = coords[0,:,0], coords[:,0,1]
    return x, y


class NamLoader:
    '''A class to download, subsets, and cache NAM forecasts.

    A `NamLoader` downloads NAM-NMM forecasts from NOAA, subsets their features
    and geographic scope, converts the data to netCDF, and caches the result.

    TODO:
        - describe the data format
    '''
    class CacheMiss(Exception): pass

    def __init__(self,
            cache_dir='./data/NAM-NMM',
            fail_fast=False,
            save_nc=True,
            keep_gribs=False,
            parallel=False):
        '''Creates a loader for NAM data.

        Args:
            cache_dir (Path or str):
                The path to the cache.
            fail_fast (bool):
                If true, the download errors are treated as fatal.
                Otherwise downloads are retried with exponential backoff.
                This overrides the `max_tries` argument of the `download`
                method.
            save_nc (bool):
                Convert the dataset to netCDF on disk.
            keep_gribs (bool):
                Keep the GRIB files after converting to netCDF.
            parallel (bool):
                Perform download and preprocess operations in parallel.
                Note that this will speed up the I/O operations, but the
                preprocessing may be slower because of the GIL.
        '''
        self.cache_dir = Path(cache_dir)
        self.fail_fast = bool(fail_fast)
        self.save_nc = bool(save_nc)
        self.keep_gribs = bool(keep_gribs)
        self.parallel = bool(parallel)

        self.cache_dir.mkdir(exist_ok=True)

        if self.parallel:
            logger.info('Parallel loading enabled')
            self._mapper = ThreadPoolExecutor().map
        else:
            self._mapper = map

    def grib_url(self, reftime, forecast):
        '''The URL for a specific forecast.
        '''
        reftime = normalize_reftime(reftime)
        now = np.datetime64('now')
        delta = now - reftime
        if delta > np.timedelta64(7, 'D'):
            if reftime < np.datetime64('2017-04-01'):
                url_fmt = ARCHIVE_URL_1
            else:
                url_fmt = ARCHIVE_URL_2
        else:
            url_fmt = PROD_URL
        return url_fmt.format(ref=reftime.astype(object), forecast=forecast)

    def grib_path(self, reftime, forecast):
        '''The path for a forecast GRIB.
        '''
        reftime = normalize_reftime(reftime).astype(object)
        prefix_fmt = 'nam.{ref.year:04d}{ref.month:02d}{ref.day:02d}'
        filename_fmt = 'nam.t{ref.hour:02d}z.awphys{forecast:02d}.tm00.grib'
        prefix = prefix_fmt.format(forecast=forecast, ref=reftime)
        filename = filename_fmt.format(forecast=forecast, ref=reftime)
        return self.cache_dir / prefix / filename

    def nc_path(self, reftime):
        '''The path to the netCDF cache for a reference time.
        '''
        reftime = normalize_reftime(reftime).astype(object)
        prefix = f'nam.{reftime.year:04d}{reftime.month:02d}{reftime.day:02d}'
        filename = f'nam.t{reftime.hour:02d}z.awphys.tm00.nc'
        return self.cache_dir / prefix / filename

    def download(self, reftime, forecast, max_tries=8, timeout=10):
        '''Download the GRIB files for this dataset.

        Args:
            reftime (datetime64):
                The reference time to download.
            forecast (int):
                The forecast hour to download
            max_tries (int):
                The maximum number of failed downloads for a single file
                before raising an `IOError`. This option is ignored if
                `fail_fast` is set on the NamLoader.
            timeout (int):
                The network timeout in seconds.
                Note that the government servers can be kinda slow.
        '''
        if self.fail_fast:
            max_tries = 1

        url = self.grib_url(reftime, forecast)
        path = self.grib_path(reftime, forecast)
        path.parent.mkdir(exist_ok=True)

        if path.exists():
            return

        for i in range(max_tries):
            try:
                # Perform a streaming download because the files are big.
                logger.info(f'downloading {url}')
                with path.open('wb') as fd:
                    r = requests.get(url, timeout=timeout, stream=True)
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=128):
                        fd.write(chunk)
                break

            except IOError as err:
                # IOError includes both system and HTTP errors.
                # Retry with exponential backoff.
                logger.warning(err)
                path.unlink()
                if i + 1 == max_tries:
                    logger.error(f'download of {path.name} failed, giving up')
                    raise err
                else:
                    delay = 2**i
                    logger.warning(f'download of {path.name} failed, retrying in {delay}s')
                    sleep(delay)
                    continue

            except (Exception, SystemExit, KeyboardInterrupt) as err:
                # Partial files will break future downloads, must delete.
                # SystemExit and KeyboardInterrupt must be caught explicitly.
                path.unlink()
                raise err

    def load_grib(self, reftime, forecast):
        '''Load a forecast from GRIB.
        '''
        self.download(reftime, forecast)

        reftime = normalize_reftime(reftime)
        path = self.grib_path(reftime, forecast)
        logger.info(f'reading {path}')

        ds = xr.open_dataset(path, engine='pynio')

        # Normalize the first format.
        # This format occurs when reading a GRIB1 file with xarray and the PyNIO backend.
        # Note `gridx_218` is the y coordinate, likewise `gridy_218` is the x coordinate.
        if 'gridx_218' in ds.dims:
            features = {
                'DLWRF_218_SFC':  'DLWRF_SFC',              'DSWRF_218_SFC':  'DSWRF_SFC',
                'PRES_218_SFC':   'PRES_SFC',
                'PRES_218_MWSL':  'PRES_MWSL',              'PRES_218_TRO':   'PRES_TRO',
                'T_CDC_218_EATM': 'TCC_EATM',               'TMP_218_SPDY':   'TMP_SPDY',
                'TMP_218_SFC':    'TMP_SFC',                'TMP_218_ISBL':   'TMP_ISBL',
                'TMP_218_HTGL':   'TMP_HTGL',               'TMP_218_TRO':    'TMP_TRO',
                'R_H_218_SIGY':   'RH_SIGY',                'R_H_218_SPDY':   'RH_SPDY',
                'R_H_218_ISBL':   'RH_ISBL',
                'R_H_218_0DEG':   'RH_0DEG',                'U_GRD_218_SPDY': 'UGRD_SPDY',
                'U_GRD_218_ISBL': 'UGRD_ISBL',              'U_GRD_218_HTGL': 'UGRD_HTGL',
                'U_GRD_218_220':  'UGRD_TOA',               'U_GRD_218_MWSL': 'UGRD_MWSL',
                'U_GRD_218_TRO':  'UGRD_TRO',               'V_GRD_218_SPDY': 'VGRD_SPDY',
                'V_GRD_218_ISBL': 'VGRD_ISBL',              'V_GRD_218_HTGL': 'VGRD_HTGL',
                'V_GRD_218_220':  'VGRD_TOA',               'V_GRD_218_MWSL': 'VGRD_MWSL',
                'V_GRD_218_TRO':  'VGRD_TRO',               'VIS_218_SFC':    'VIS_SFC',
                'LHTFL_218_SFC':  'LHTFL_SFC',              'SHTFL_218_SFC':  'SHTFL_SFC',
                'REFC_218_EATM':  'REFC_EATM',              'REFD_218_HTGL':  'REFD_HTGL',
                'REFD_218_HYBL':  'REFD_HYBL',              'V_VEL_218_ISBL': 'VVEL_ISBL',
                'HGT_218_SFC':    'HGT_SFC',                'HGT_218_ISBL':   'HGT_ISBL',
                'HGT_218_CBL':    'HGT_CBL',                'HGT_218_220':    'HGT_TOA',
                'HGT_218_LLTW':   'HGT_LLTW',               'HGT_218_0DEG':   'HGT_0DEG',
                'P_WAT_218_EATM': 'PWAT_EATM',              'TKE_218_ISBL':   'TKE_ISBL',

                'lv_HTGL3':       'z_HTGL1',                'lv_HTGL5':       'z_HTGL2',
                'lv_HTGL9':       'z_HTGL3',                'lv_ISBL2':       'z_ISBL',
                'lv_SPDY4':       'z_SPDY',
                'gridx_218':      'y',                      'gridy_218':      'x',
                'gridlat_218':    'lat',                    'gridlon_218':    'lon',
            }
            unwanted = [k for k in ds.keys() if k not in features]
            ds = ds.drop(unwanted)
            ds = ds.rename(features)
            ds['z_ISBL'].data *= 100
            ds['z_ISBL'].attrs['units'] = 'Pa'

        # Normalize the second format.
        # This format occurs after the change from GRIB1 to GRIB2 (circa April 2017).
        else:
            features = {
                'DLWRF_P0_L1_GLC0':   'DLWRF_SFC',          'DSWRF_P0_L1_GLC0':   'DSWRF_SFC',
                'PRES_P0_L1_GLC0':    'PRES_SFC',
                'PRES_P0_L6_GLC0':    'PRES_MWSL',          'PRES_P0_L7_GLC0':    'PRES_TRO',
                'TCDC_P0_L200_GLC0':  'TCC_EATM',           'TMP_P0_2L108_GLC0':  'TMP_SPDY',
                'TMP_P0_L1_GLC0':     'TMP_SFC',            'TMP_P0_L100_GLC0':   'TMP_ISBL',
                'TMP_P0_L103_GLC0':   'TMP_HTGL',           'TMP_P0_L7_GLC0':     'TMP_TRO',
                'RH_P0_2L104_GLC0':   'RH_SIGY',            'RH_P0_2L108_GLC0':   'RH_SPDY',
                'RH_P0_L100_GLC0':    'RH_ISBL',
                'RH_P0_L4_GLC0':      'RH_0DEG',            'UGRD_P0_2L108_GLC0': 'UGRD_SPDY',
                'UGRD_P0_L100_GLC0':  'UGRD_ISBL',          'UGRD_P0_L103_GLC0':  'UGRD_HTGL',
                'UGRD_P0_L220_GLC0':  'UGRD_TOA',           'UGRD_P0_L6_GLC0':    'UGRD_MWSL',
                'UGRD_P0_L7_GLC0':    'UGRD_TRO',           'VGRD_P0_2L108_GLC0': 'VGRD_SPDY',
                'VGRD_P0_L100_GLC0':  'VGRD_ISBL',          'VGRD_P0_L103_GLC0':  'VGRD_HTGL',
                'VGRD_P0_L220_GLC0':  'VGRD_TOA',           'VGRD_P0_L6_GLC0':    'VGRD_MWSL',
                'VGRD_P0_L7_GLC0':    'VGRD_TRO',           'VIS_P0_L1_GLC0':     'VIS_SFC',
                'LHTFL_P0_L1_GLC0':   'LHTFL_SFC',          'SHTFL_P0_L1_GLC0':   'SHTFL_SFC',
                'REFC_P0_L200_GLC0':  'REFC_EATM',          'REFD_P0_L103_GLC0':  'REFD_HTGL',
                'REFD_P0_L105_GLC0':  'REFD_HYBL',          'VVEL_P0_L100_GLC0':  'VVEL_ISBL',
                'HGT_P0_L1_GLC0':     'HGT_SFC',            'HGT_P0_L100_GLC0':   'HGT_ISBL',
                'HGT_P0_L2_GLC0':     'HGT_CBL',            'HGT_P0_L220_GLC0':   'HGT_TOA',
                'HGT_P0_L245_GLC0':   'HGT_LLTW',           'HGT_P0_L4_GLC0':     'HGT_0DEG',
                'PWAT_P0_L200_GLC0':  'PWAT_EATM',          'TKE_P0_L100_GLC0':   'TKE_ISBL',

                'lv_HTGL1':           'z_HTGL1',            'lv_HTGL3':           'z_HTGL2',
                'lv_HTGL6':           'z_HTGL3',            'lv_ISBL0':           'z_ISBL',
                'lv_SPDL2':           'z_SPDY',
                'xgrid_0':            'x',                  'ygrid_0':            'y',
                'gridlat_0':          'lat',                'gridlon_0':          'lon',
            }
            unwanted = [k for k in ds.keys() if k not in features]
            ds = ds.drop(unwanted)
            ds = ds.rename(features)

        # Subset the geographic region to a square area centered around Macon, GA.
        # The slice was computed roughly like this:
        # >>> a = (24.415094, -93.995674) # latlon coordinates
        # >>> b = (40.090744, -71.77018) # latlon coordinates
        # >>> diagonal_slice((ds.lat, ds.lon), a, b)
        ds = ds.isel(y=slice(63, 223, None), x=slice(355, 515, None))

        # Free memory from unused features and areas.
        ds = ds.copy(deep=True)

        # Compute the coordinates for x and y
        x, y = proj_coords(ds.lat.data, ds.lon.data)
        ds = ds.assign_coords(x=x, y=y)

        # Add a z dimension to variables that don't have one.
        for v in ds.data_vars:
            if ds[v].dims == ('y', 'x'):
                layer = ds[v].name.split('_')[1]
                ds[v] = ds[v].expand_dims(f'z_{layer}')

        # Create reftime and forecast dimensions.
        # Both are stored as integers with appropriate units.
        ds = ds.assign_coords(
            reftime=reftime.astype('datetime64[h]').astype('int'),
            forecast=forecast,
        )
        for v in ds.data_vars:
            ds[v] = ds[v].expand_dims(('reftime', 'forecast'))

        # Fix the z_SPDY coordinate.
        # The layer is defined in term of bounds above and below.
        # The dataset expresses this as three coordinates: the index, lower bound, and upper bound.
        # We kept the index and now replace the values to be the upper bound, in Pascals
        ds['z_SPDY'] = ds['z_SPDY'].assign_attrs(
            comment='The values give the upper bound of the layer, the lower bound is 3000 Pa less',
        )
        ds['z_SPDY'].data = np.array([3000, 6000, 9000, 12000, 15000, 18000])

        # Set metadata according to CF conventions
        # http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/cf-conventions.html
        metadata = {
            # Data Variables
            # TODO: The wind directions may be backwards, should be confirmed with NCEP.
            'DLWRF_SFC': {'standard_name':'downwelling_longwave_flux',         'units':'W/m^2'},
            'DSWRF_SFC': {'standard_name':'downwelling_shortwave_flux',        'units':'W/m^2'},
            'HGT_0DEG':  {'standard_name':'geopotential_height',               'units':'gpm'},
            'HGT_CBL':   {'standard_name':'geopotential_height',               'units':'gpm'},
            'HGT_ISBL':  {'standard_name':'geopotential_height',               'units':'gpm'},
            'HGT_LLTW':  {'standard_name':'geopotential_height',               'units':'gpm'},
            'HGT_TOA':   {'standard_name':'geopotential_height',               'units':'gpm'},
            'HGT_SFC':   {'standard_name':'geopotential_height',               'units':'gpm'},
            'PRES_MWSL': {'standard_name':'air_pressure',                      'units':'Pa'},
            'PRES_SFC':  {'standard_name':'air_pressure',                      'units':'Pa'},
            'PRES_TRO':  {'standard_name':'air_pressure',                      'units':'Pa'},
            'PWAT_EATM': {'standard_name':'atmosphere_water_vapor_content',    'units':'kg/m^2'},
            'REFC_EATM': {'standard_name':'equivalent_reflectivity_factor',    'units':'dBZ'},
            'REFD_HTGL': {'standard_name':'equivalent_reflectivity_factor',    'units':'dBZ'},
            'REFD_HYBL': {'standard_name':'equivalent_reflectivity_factor',    'units':'dBZ'},
            'RH_0DEG':   {'standard_name':'relative_humidity',                 'units':'%'},
            'RH_ISBL':   {'standard_name':'relative_humidity',                 'units':'%'},
            'RH_SIGY':   {'standard_name':'relative_humidity',                 'units':'%'},
            'RH_SPDY':   {'standard_name':'relative_humidity',                 'units':'%'},
            'LHTFL_SFC': {'standard_name':'upward_latent_heat_flux',           'units':'W/m2'},
            'SHTFL_SFC': {'standard_name':'upward_sensible_heat_flux',         'units':'W/m2'},
            'TCC_EATM':  {'standard_name':'cloud_area_fraction',               'units':'%'},
            'TKE_ISBL':  {'standard_name':'atmosphere_kinetic_energy_content', 'units':'J/kg'},
            'TMP_HTGL':  {'standard_name':'air_temperature',                   'units':'K'},
            'TMP_ISBL':  {'standard_name':'air_temperature',                   'units':'K'},
            'TMP_SFC':   {'standard_name':'air_temperature',                   'units':'K'},
            'TMP_SPDY':  {'standard_name':'air_temperature',                   'units':'K'},
            'TMP_TRO':   {'standard_name':'air_temperature',                   'units':'K'},
            'UGRD_HTGL': {'standard_name':'eastward_wind',                     'units':'m/s'},
            'UGRD_ISBL': {'standard_name':'eastward_wind',                     'units':'m/s'},
            'UGRD_MWSL': {'standard_name':'eastward_wind',                     'units':'m/s'},
            'UGRD_TOA':  {'standard_name':'eastward_wind',                     'units':'m/s'},
            'UGRD_SPDY': {'standard_name':'eastward_wind',                     'units':'m/s'},
            'UGRD_TRO':  {'standard_name':'eastward_wind',                     'units':'m/s'},
            'VGRD_HTGL': {'standard_name':'northward_wind',                    'units':'m/s'},
            'VGRD_ISBL': {'standard_name':'northward_wind',                    'units':'m/s'},
            'VGRD_MWSL': {'standard_name':'northward_wind',                    'units':'m/s'},
            'VGRD_TOA':  {'standard_name':'northward_wind',                    'units':'m/s'},
            'VGRD_SPDY': {'standard_name':'northward_wind',                    'units':'m/s'},
            'VGRD_TRO':  {'standard_name':'northward_wind',                    'units':'m/s'},
            'VIS_SFC':   {'standard_name':'visibility',                        'units':'m'},
            'VVEL_ISBL': {'standard_name':'vertical_air_velocity_expressed_as_tendency_of_pressure',
                          'units':'Pa/s'},

            # Coordinates
            # I couldn't find standard names for all of the layers...
            # I'm not sure if both forecast and reftime should be marked as axis T...
            'x':        {'axis':'X', 'standard_name':'projection_x_coordinate',   'units':'m'},
            'y':        {'axis':'Y', 'standard_name':'projection_y_coordinate',   'units':'m'},
            'z_CBL':    {'axis':'Z', 'standard_name':'cloud_base'},
            'z_HYBL':   {'axis':'Z', 'standard_name':'atmosphere_hybrid_sigma_pressure_coordinate'},
            'z_TOA':    {'axis':'Z', 'standard_name':'toa'},
            'z_SFC':    {'axis':'Z', 'standard_name':'surface'},
            'z_SIGY':   {'axis':'Z', 'standard_name':'atmosphere_sigma_coordinate'},
            'z_TRO':    {'axis':'Z', 'standard_name':'tropopause'},
            'z_SPDY':   {'axis':'Z', 'long_name':'specified pressure difference', 'units':'Pa'},
            'z_HTGL1':  {'axis':'Z', 'long_name':'fixed_height_above_ground',     'units':'m'},
            'z_HTGL2':  {'axis':'Z', 'long_name':'fixed_height_above_ground',     'units':'m'},
            'z_HTGL3':  {'axis':'Z', 'long_name':'fixed_height_above_ground',     'units':'m'},
            'z_ISBL':   {'axis':'Z', 'long_name':'isobaric_level',                'units':'Pa'},
            'z_0DEG':   {'axis':'Z', 'long_name':'0_degree_C_isotherm'},
            'z_EATM':   {'axis':'Z', 'long_name':'entire_atmosphere'},
            'z_LLTW':   {'axis':'Z', 'long_name':'lowest_level_of_the_wet_bulb_zero'},
            'z_MWSL':   {'axis':'Z', 'long_name':'max_wind_surface_layer'},
            'forecast': {'axis':'T', 'standard_name':'forecast_period',           'units':'hours'},
            'reftime':  {'axis':'T', 'standard_name':'forecast_reference_time',   'units':'hours since 1970-01-01T00:00'},
            'lat':      {'standard_name':'latitude',  'units':'degree_north'},
            'lon':      {'standard_name':'longitude', 'units':'degree_east'},
        }
        for v in metadata:
            ds[v] = ds[v].assign_attrs(metadata[v])
        ds = ds.assign_attrs(
            title='NAM-UGA, a subset of NAM-NMM for solar forecasting research in Georgia',
            history=f'{np.datetime64("now")}Z Initial conversion from GRIB files released by NCEP',
        )

        ds = xr.decode_cf(ds)
        return ds

    def _combine(self, datasets):
        '''Combine a list of datasets.

        This is non-trivial because the old-format and the new-format are not
        perfectly aligned; the x and y coordinates are offset by up to 4 km.

        We force all to align to the final dataset's spatial coordinates.
        '''
        coords = {
            'x':   datasets[-1].x,
            'y':   datasets[-1].y,
            'lat': datasets[-1].lat,
            'lon': datasets[-1].lon,
        }
        datasets = (ds.drop(('x', 'y', 'lat', 'lon')) for ds in datasets)
        logger.info('joining datasets')
        ds = xr.concat(datasets, dim='reftime')
        ds = ds.assign_coords(**coords)
        return ds

    def open_gribs(self, reftime='now'):
        '''Load the forecasts from GRIB, downlading if they do not exist.

        Args:
            reftime (datetime64):
                The reference time to open.

        Returns:
            An `xr.Dataset` describing this forecast.
        '''
        datasets = self._mapper(lambda f: self.load_grib(reftime, f), FORECAST_PERIOD)
        ds = xr.concat(datasets, dim='forecast')

        if self.save_nc:
            path = self.nc_path(reftime)
            logger.info(f'writing {path}')
            ds.to_netcdf(str(path)) # must be str, can't be Path, should fix in xarray
            if not self.keep_gribs:
                logger.info('deleting local gribs')
                for forecast in FORECAST_PERIOD:
                    path = self.grib_path(reftime, forecast)
                    path.unlink()

        return ds

    def open_nc(self, reftime='now'):
        '''Load the forecasts from a netCDF in the cache.

        Args:
            reftime (datetime64):
                The reference time to open.

        Returns:
            An `xr.Dataset` describing this forecast.
        '''
        path = self.nc_path(reftime)
        if path.exists():
            logger.info(f'reading {path}')
            ds = xr.open_dataset(
                path,
                autoclose=True,
                chunks={},
            )
            return ds
        else:
            raise NamLoader.CacheMiss(reftime)

    def open(self, *reftimes):
        '''Load and combine forecasts for some reference times,
        downloading and preprocessing GRIBs as necessary.

        If the dataset exists as a local netCDF file, it is loaded and
        returned. Otherwise, any missing GRIB files are downloaded and
        preprocessed into an xarray Dataset. The dataset is then saved as a
        netCDF file, the GRIBs are deleted, and the dataset is returned.

        Args:
            reftimes (datetime64):
                The reference times to open.

        Returns (xr.Dataset):
            Returns a single dataset containing all forecasts at the given
            reference times. Some data may be dropped when combining forecasts.
        '''
        def _open(reftime):
            try:
                ds = self.open_nc(reftime)
            except NamLoader.CacheMiss as e:
                ds = self.open_gribs(reftime)
            return ds

        if len(reftimes) == 0:
            return _open('now')
        else:
            datasets = [_open(r) for r in reftimes]
            return self._combine(datasets)

    def open_range(self, start='2017-01-01', stop='today'):
        '''Load and combine forecasts for a range of reference times.

        NOTE: This method only loads data from the cache.

        Args:
            start (datetime64):
                The first time in the range.
                The default is 2017-01-01T00:00
            stop (datetime64):
                The last time in the range.
                The default is the start of the current day.

        Returns (xr.Dataset):
            Returns a single dataset containing all forecasts at the given
            reference times. Some data may be dropped when combining forecasts.
        '''
        start = normalize_reftime(start)
        stop = normalize_reftime(stop)

        datasets = []
        delta = np.timedelta64(6, 'h')
        while start < stop:
            try:
                ds = self.open_nc(start)
                datasets.append(ds)
            except OSError as e:
                logger.warn(f'error reading forecast for {start}')
                logger.warn(e)
            except NamLoader.CacheMiss:
                pass
            start += delta

        return self._combine(datasets)


@xr.register_dataarray_accessor('geo')
class GeoExtension:
    '''Extends `xr.DataArray` with geographic specific features.
    '''
    def __init__(self, xr_dataarray):
        self.x = xr_dataarray

    def plot(self, scale='10m', show=True, block=False):
        '''A helper to plot geographic data.

        Args:
            scale (str):
                The resolution of the coastlines and state/country borders.
                Must be one of '10m' (highest resolution), '50m', or '110m'.
            show (bool):
                If True, show the plot immediately.
            block (bool):
                The blocking behavior when showing the plot.

        Example:
            Plot the 0-hour forecast of surface temperature:
            >>> plot_geo(ds.isel(reftime=0, forecast=0).TMP_SFC)
        '''
        vmin = self.x.min()
        vmax = self.x.max()

        x = self.x
        while x.ndim > 2:
            x = x[0]

        feature_kws = {'scale':scale, 'facecolor':'none', 'edgecolor':'black'}
        coast = cf.NaturalEarthFeature('physical', 'coastline', **feature_kws)
        countries = cf.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land', **feature_kws)
        states = cf.NaturalEarthFeature('cultural', 'admin_1_states_provinces_lines', **feature_kws)

        fig = plt.figure()
        ax = plt.axes(projection=NAM218_PROJ)
        ax.add_feature(coast)
        ax.add_feature(countries)
        ax.add_feature(states)
        im = ax.pcolormesh(x.x, x.y, x.data, vmin=vmin, vmax=vmax)

        if show:
            plt.show(block=block)
        return ax
