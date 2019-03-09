'''Provides access to a subset of the NAM-NMM dataset.

    The North American Mesoscale Forecast System (NAM) is one of the
    major weather models run by the National Centers for Environmental
    Prediction (NCEP) for producing weather forecasts. Dozens of weather
    parameters are available from the NAM grids, from temperature and
    precipitation to lightning and turbulent kinetic energy.

    As of June 20, 2006, the NAM model has been running with a non-
    hydrostatic version of the Weather Research and Forecasting (WRF)
    model at its core. This version of the NAM is also known as the NAM
    Non-hydrostatic Mesoscale Model (NAM-NMM).

Most users will be interested in the :func:`open`, :func:`open_local`, and
:func:`open_range` functions which select forecasts by reference time. The
actual data loading logic is encapsulated in the :class:`NamLoader` class.

The dataset live remotely. A live feed is provided by NCEP and an 11
month archive is provided by NCDC (both are divisions of NOAA). This
module caches the data locally, allowing us to build a larger archive.
The remote dataset is provided in GRIB format, while this module uses
the netCDF format for its local storage.

This module provides access to only a subset of the NAM-NMM dataset.
The geographic region is reduced and centered around Georgia, and only
as subset of the variables are provided.

Queries return an instance of :class:`xarray.Dataset` where each variable
has exactly five dimensions: reftime, forecast, z, y, and x. The z-axis
for each variable has a different name depending on the type of index
measuring the axis, e.g. ``z_ISBL`` for isobaric pressure levels.
'''

import logging
from functools import reduce
from pathlib import Path
from time import sleep

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import requests
import xarray as xr

from apollo import storage


# Module level logger
logger = logging.getLogger(__name__)


# URLs of remote grib files.
# PROD_URL typically has the most recent 7 days.
# ARCHIVE_URL typically has the most recent 11 months, about 1 week behind.
PROD_URL = 'http://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.{ref.year:04d}{ref.month:02d}{ref.day:02d}/nam.t{ref.hour:02d}z.awphys{forecast:02d}.tm00.grib2'
ARCHIVE_URL = 'https://nomads.ncdc.noaa.gov/data/meso-eta-hi/{ref.year:04d}{ref.month:02d}/{ref.year:04d}{ref.month:02d}{ref.day:02d}/nam_218_{ref.year:04d}{ref.month:02d}{ref.day:02d}_{ref.hour:02d}00_{forecast:03d}.grb2'


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
    preprocessed into an :class:`xarray.Dataset`. The dataset is then
    saved as a netCDF file, the GRIBs are deleted, and the dataset is
    returned.

    Arguments:
        *reftimes (pandas.Timestamp or numpy.datetime64 or datetime.datetime or str):
            The reference times to open. If none are given, the current
            forecast period is used.
        **kwargs:
            Forwarded to :class:`NamLoader`.

    Returns:
        xarray.Dataset:
            Returns a single dataset containing all forecasts at the given
            reference times. Some data may be dropped when combining forecasts.
    '''
    loader = NamLoader(**kwargs)
    return loader.open(*reftimes)


def open_range(start, stop, **kwargs):
    '''Load and combine forecasts for a range of reference times.

    NOTE: This method only loads data from the local store.

    Arguments:
        start (pandas.Timestamp or numpy.datetime64 or datetime.datetime or str):
            The first time in the range.
        stop (pandas.Timestamp or numpy.datetime64 or datetime.datetime or str):
            The last time in the range.
        **kwargs:
            Forwarded to :class:`NamLoader`.

    Returns:
        xarray.Dataset:
            Returns a single dataset containing all forecasts at the given
            reference times. Some data may be dropped when combining forecasts.
    '''
    loader = NamLoader(**kwargs)
    return loader.open_range(start, stop)

def open_local(reftime='now', **kwargs):
    '''Load a forecast from netCDF in the local store.

    Arguments:
        reftime (pandas.Timestamp or numpy.datetime64 or datetime.datetime or str):
            The reference time to open.
        **kwargs:
            Forwarded to :class:`NamLoader`.

    Returns:
        ds (xarray.Dataset):
            A dataset for the forecast at this reftime.

    Raises:
        CacheMiss:
            An exception is raised if the forecast does not exist locally.
    '''
    loader = NamLoader(**kwargs)
    return loader.open_local(reftime)


def proj_coords(lats, lons):
    '''Transform geographic coordinates into the NAM218 projection.
    The input is a geographic area described by a pair of 2D arrays giving the
    latitude and longitude of each cell. The input must map to a square area in
    the projected coordinates. The values returned are 1D indices along both
    the x and y axes.

    The output is undefined if the input does not map to a square area.

    Arguments:
        lats (numpy.ndarray):
            The latitude at each cell as a 2D array.
        lons (numpy.ndarray):
            The longitude at each cell as a 2D array.
    Returns:
        x (numpy.ndarray):
            The coordinates for the x axis in meters as a 1D array.
        y (numpy.ndarray):
            The coordinates for the y axis in meters as a 1D array.
    '''
    unproj = ccrs.PlateCarree()
    coords = NAM218_PROJ.transform_points(unproj, lons, lats)
    x, y = coords[0,:,0], coords[:,0,1]
    return x, y


class CacheMiss(Exception):
    '''A requested forecast does not exist in the local store.
    '''
    pass


class NamLoader:
    '''A class to download, subsets, and store NAM forecasts.

    A `NamLoader` downloads NAM-NMM forecasts from NOAA, subsets their features
    and geographic scope, converts the data to netCDF, and stores the result.
    '''

    def __init__(self, fail_fast=False, save_nc=True, keep_gribs=False):
        '''Creates a loader for NAM data.

        Arguments:
            fail_fast (bool):
                If true, the download errors are treated as fatal.
                Otherwise downloads are retried with exponential backoff.
                This overrides the `max_tries` argument of the `download`
                method.
            save_nc (bool):
                Convert the dataset to netCDF on disk.
            keep_gribs (bool):
                Keep the GRIB files after converting to netCDF.
        '''
        self.fail_fast = bool(fail_fast)
        self.save_nc = bool(save_nc)
        self.keep_gribs = bool(keep_gribs)

        self.data_dir = storage.get('NAM-NMM')

    def grib_url(self, reftime, forecast):
        '''The URL for a specific forecast.

        Arguments:
            reftime (pandas.Timestamp or numpy.datetime64 or datetime.datetime or str):
                The reference time.
            forecast (int):
                The forecast hour.

        Returns:
            str:
                A URL to a GRIB file.
        '''
        reftime = pd.Timestamp(reftime).floor('6h')
        now = pd.Timestamp('now').floor('6h')
        delta = now - reftime
        if pd.Timedelta(7, 'd') < delta:
            url_fmt = ARCHIVE_URL
        else:
            url_fmt = PROD_URL
        return url_fmt.format(ref=reftime, forecast=forecast)

    def grib_path(self, reftime, forecast):
        '''The path for a forecast GRIB.

        Arguments:
            reftime (pandas.Timestamp or numpy.datetime64 or datetime.datetime or str):
                The reference time.
            forecast (int):
                The forecast hour.

        Returns:
            pathlib.Path:
                The local path for a GRIB file, which may not exist.
        '''
        reftime = pd.Timestamp(reftime).floor('6h')
        prefix_fmt = 'nam.{ref.year:04d}{ref.month:02d}{ref.day:02d}'
        filename_fmt = 'nam.t{ref.hour:02d}z.awphys{forecast:02d}.tm00.grib'
        prefix = prefix_fmt.format(forecast=forecast, ref=reftime)
        filename = filename_fmt.format(forecast=forecast, ref=reftime)
        return self.data_dir / prefix / filename

    def nc_path(self, reftime):
        '''The path to the netCDF forecast for the given reference time.

        Arguments:
            reftime (pandas.Timestamp or numpy.datetime64 or datetime.datetime or str):
                The reference time.

        Returns:
            pathlib.Path:
                The local path to a netCDF file, which may not exist.
        '''
        reftime = reftime = pd.Timestamp(reftime).floor('6h')
        prefix = f'nam.{reftime.year:04d}{reftime.month:02d}{reftime.day:02d}'
        filename = f'nam.t{reftime.hour:02d}z.awphys.tm00.nc'
        return self.data_dir / prefix / filename

    def download(self, reftime, forecast, max_tries=8, timeout=10):
        '''Ensure that the GRIB for this reftime and forecast exists locally.

        Arguments:
            reftime (pandas.Timestamp or numpy.datetime64 or datetime.datetime or str):
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

        Returns:
            path (pathlib.Path):
                The path to the downloaded GRIB.
        '''
        if self.fail_fast:
            max_tries = 1

        url = self.grib_url(reftime, forecast)
        path = self.grib_path(reftime, forecast)
        path.parent.mkdir(exist_ok=True)

        # TODO: We need a better heuristic than skipping if the path exists.
        if path.exists():
            return path

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
                # Partial files should be deleted
                # SystemExit and KeyboardInterrupt must be caught explicitly.
                path.unlink()
                raise err

        return path

    def _process_grib(self, ds, reftime, forecast):
        '''Process a forecast loaded from GRIB.

        GRIB files contain a forecast for a specific forecast hour at a specific
        reftime, including all NAM data variables for the entire NAM 218 grid.

        This method trims the dataset to the subset of variables and geographic
        region that we are interested in, normalizes variable names and shapes
        to a more consistent format, and adds additional metadata.

        Arguments:
            ds (xarray.Dataset):
                The dataset to process.
            reftime (pandas.Timestamp or numpy.datetime64 or datetime.datetime or str):
                The reference time associated with the dataset.
            forecast (int):
                The forecast hour associated with the dataset.

        Returns:
            xarray.Dataset:
                A processed dataset.
        '''
        features = {
            # Data variables
            'DLWRF_P0_L1_GLC0':  'DLWRF_SFC',    'DSWRF_P0_L1_GLC0':   'DSWRF_SFC',
            'PRES_P0_L1_GLC0':   'PRES_SFC',
            'PRES_P0_L6_GLC0':   'PRES_MWSL',    'PRES_P0_L7_GLC0':    'PRES_TRO',
            'TCDC_P0_L200_GLC0': 'TCC_EATM',     'TMP_P0_2L108_GLC0':  'TMP_SPDY',
            'TMP_P0_L1_GLC0':    'TMP_SFC',      'TMP_P0_L100_GLC0':   'TMP_ISBL',
            'TMP_P0_L103_GLC0':  'TMP_HTGL',     'TMP_P0_L7_GLC0':     'TMP_TRO',
            'RH_P0_2L104_GLC0':  'RH_SIGY',      'RH_P0_2L108_GLC0':   'RH_SPDY',
            'RH_P0_L100_GLC0':   'RH_ISBL',
            'RH_P0_L4_GLC0':     'RH_0DEG',      'UGRD_P0_2L108_GLC0': 'UGRD_SPDY',
            'UGRD_P0_L100_GLC0': 'UGRD_ISBL',    'UGRD_P0_L103_GLC0':  'UGRD_HTGL',
            'UGRD_P0_L220_GLC0': 'UGRD_TOA',     'UGRD_P0_L6_GLC0':    'UGRD_MWSL',
            'UGRD_P0_L7_GLC0':   'UGRD_TRO',     'VGRD_P0_2L108_GLC0': 'VGRD_SPDY',
            'VGRD_P0_L100_GLC0': 'VGRD_ISBL',    'VGRD_P0_L103_GLC0':  'VGRD_HTGL',
            'VGRD_P0_L220_GLC0': 'VGRD_TOA',     'VGRD_P0_L6_GLC0':    'VGRD_MWSL',
            'VGRD_P0_L7_GLC0':   'VGRD_TRO',     'VIS_P0_L1_GLC0':     'VIS_SFC',
            'LHTFL_P0_L1_GLC0':  'LHTFL_SFC',    'SHTFL_P0_L1_GLC0':   'SHTFL_SFC',
            'REFC_P0_L200_GLC0': 'REFC_EATM',    'REFD_P0_L103_GLC0':  'REFD_HTGL',
            'REFD_P0_L105_GLC0': 'REFD_HYBL',    'VVEL_P0_L100_GLC0':  'VVEL_ISBL',
            'HGT_P0_L1_GLC0':    'HGT_SFC',      'HGT_P0_L100_GLC0':   'HGT_ISBL',
            'HGT_P0_L2_GLC0':    'HGT_CBL',      'HGT_P0_L220_GLC0':   'HGT_TOA',
            'HGT_P0_L245_GLC0':  'HGT_LLTW',     'HGT_P0_L4_GLC0':     'HGT_0DEG',
            'PWAT_P0_L200_GLC0': 'PWAT_EATM',    'TKE_P0_L100_GLC0':   'TKE_ISBL',

            # Coordinate variables
            'lv_HTGL1':  'z_HTGL1',    'lv_HTGL3':  'z_HTGL2',
            'lv_HTGL6':  'z_HTGL3',    'lv_ISBL0':  'z_ISBL',
            'lv_SPDL2':  'z_SPDY',
            'xgrid_0':   'x',          'ygrid_0':   'y',
            'gridlat_0': 'lat',        'gridlon_0': 'lon',
        }
        unwanted = [k for k in ds.variables.keys() if k not in features]
        ds = ds.drop(unwanted)
        ds = ds.rename(features)

        # Subset the geographic region to a square area centered around Macon, GA.
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
        # The reftime dimension is hours since the Unix epoch (1970-01-01 00:00).
        # The forecast dimension is hours since the reftime.
        reftime = pd.Timestamp(reftime).floor('6h')
        epoch = pd.Timestamp('1970-01-01 00:00')
        delta_seconds = int((reftime - epoch).total_seconds())
        delta_hours = delta_seconds // 60 // 60
        ds = ds.assign_coords(
            reftime=delta_hours,
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

        now = pd.Timestamp("now", tz='utc')
        ds.attrs['title'] = 'NAM-UGA, a subset of NAM-NMM for solar forecasting research in Georgia'
        ds.attrs['history'] = f'{now.isoformat()} Initial conversion from GRIB files released by NCEP\n'

        ds = xr.decode_cf(ds)
        return ds

    def open_remote(self, reftime='now'):
        '''Load a forecast from GRIBs, downlading if they do not exist.

        If the loader was initialized with ``save_nc=True`` and
        ``keep_gribs=False`` (the default), then the resulting forecast is
        converted to a netCDF file on disk and the GRIB files are discarded.

        Arguments:
            reftime (pandas.Timestamp or numpy.datetime64 or datetime.datetime or str):
                The reference time to open.

        Returns:
            xarray.Dataset:
                A dataset for the forecast at this reftime.
        '''
        datasets = []
        for forecast in FORECAST_PERIOD:
            path = self.download(reftime, forecast)
            logger.info(f'reading {path}')
            ds = xr.open_dataset(path, engine='pynio')
            ds = self._process_grib(ds, reftime, forecast)
            datasets.append(ds)
        ds = xr.concat(datasets, dim='forecast')

        if self.save_nc:
            path = self.nc_path(reftime)
            logger.info(f'writing {path}')
            ds.to_netcdf(path)
            if not self.keep_gribs:
                logger.info('deleting local gribs')
                for forecast in FORECAST_PERIOD:
                    path = self.grib_path(reftime, forecast)
                    path.unlink()
            ds = self.open_local(reftime)

        return ds

    def open_local(self, reftime='now'):
        '''Load a forecast from netCDF in the local store.

        Arguments:
            reftime (pandas.Timestamp or numpy.datetime64 or datetime.datetime or str):
                The reference time to open.

        Returns:
            ds (xarray.Dataset):
                A dataset for the forecast at this reftime.

        Raises:
            CacheMiss:
                An exception is raised if the forecast does not exist locally.
        '''
        path = self.nc_path(reftime)
        if path.exists():
            logger.info(f'reading {path}')
            ds = xr.open_dataset(path, chunks={})
            return ds
        else:
            raise CacheMiss(reftime)

    def open(self, *reftimes):
        '''Load and combine forecasts for some reference times,
        downloading and preprocessing GRIBs as necessary.

        If the dataset exists as a local netCDF file, it is loaded and
        returned. Otherwise, any missing GRIB files are downloaded and
        preprocessed into an xarray Dataset. The dataset is then saved as a
        netCDF file, the GRIBs are deleted, and the dataset is returned.

        Arguments:
            *reftimes (pandas.Timestamp or numpy.datetime64 or datetime.datetime or str):
                The reference times to open. If none are given, the current
                forecast period is used.

        Returns:
            xarray.Dataset:
                A single dataset containing all forecasts at the given reference
                times. Some data may be dropped when combining forecasts.
        '''
        def _open(reftime):
            try:
                ds = self.open_local(reftime)
            except CacheMiss:
                ds = self.open_remote(reftime)
            return ds

        if len(reftimes) == 0:
            return _open('now')
        else:
            datasets = [_open(r) for r in reftimes]
            return self._combine(datasets)

    def open_range(self, start, stop):
        '''Load and combine forecasts for a range of reference times.

        NOTE: This method only loads data from the local store.

        Arguments:
            start (pandas.Timestamp or numpy.datetime64 or datetime.datetime or str):
                The first time in the range.
            stop (pandas.Timestamp or numpy.datetime64 or datetime.datetime or str):
                The last time in the range.

        Returns:
            xarray.Dataset:
                A single dataset containing all forecasts at the given reference
                times. Some data may be dropped when combining forecasts.
        '''
        start = pd.Timestamp(start).floor('6h')
        stop = pd.Timestamp(stop).floor('6h')

        datasets = []
        delta = pd.Timedelta(6, 'h')
        while start < stop:
            try:
                ds = self.open_local(start)
                datasets.append(ds)
            except OSError as e:
                logger.warn(f'error reading forecast for {start}')
                logger.warn(e)
            except CacheMiss:
                pass
            start += delta

        return self._combine(datasets)

    def _combine(self, datasets):
        '''Combine a list of datasets.

        This is non-trivial because older datasets and the newer datasets are
        not perfectly aligned; the x and y coordinates are offset by up to 4 km.

        We force all to align to the final dataset's spatial coordinates.

        Arguments:
            datasets (Sequence[xarray.Dataset]):
                The datasets to combine.

        Returns:
            xarray.Dataset:
                The combined dataset.
        '''
        # Check for spurious coordinates that should be ignored.
        # This is due to a regression: #53
        k = set(datasets[0].coords.keys())
        coords = (set(ds.coords.keys()) for ds in datasets)
        coords = (c ^ k for c in coords)
        coords = reduce(set.union, coords)
        if len(coords) != 0:
            logger.warning('spurious coordinates detected; ignoring them')
            logger.warning(f'    {coords}')
            datasets = (ds.drop(coords) for ds in datasets)

        # Drop geographic coordinates since they might not align.
        # We use the last dataset's coordinates for the final dataset.
        tail = datasets[-1]
        coords = {'x': tail.x, 'y': tail.y, 'lat': tail.lat, 'lon': tail.lon}
        datasets = (ds.drop(('x', 'y', 'lat', 'lon')) for ds in datasets)

        logger.info('joining datasets')
        ds = xr.concat(datasets, dim='reftime')
        ds = ds.assign_coords(**coords)
        return ds
