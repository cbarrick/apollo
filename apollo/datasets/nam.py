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

The dataset lives remotely. A live feed is provided by NCEP and an 11
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
from tempfile import TemporaryDirectory

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import requests
import xarray as xr

from apollo import storage, casts


# Module level logger
logger = logging.getLogger(__name__)


# The directory containing this dataset.
DATA_DIR = storage.get('NAM-NMM')


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


def _open_mfdataset(paths):
    '''A wrapper around :func:`xarray.open_mfdataset` that applies the default
    chunking used in this module.
    '''
    return xr.open_mfdataset(
        paths = paths,
        chunks = {'x': 10, 'y': 10, 'reftime': 1, 'forecast': 12},
    )


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


def grib_url(reftime, forecast):
    '''The URL to a GRIB for the given reference and forecast times.

    This method resolves the URL from one of two sources. The production
    NAM forecasts are hosted by the National Centers for Environmental
    Prediction (NCEP, <https://www.ncep.noaa.gov/>). After seven days, the
    forecasts are moved to an eleven month archive hosted by the National
    Climatic Data Center (NCDC, <https://www.ncdc.noaa.gov/>). Older
    forecasts will resolve to the NCDC URL, but they are unlikely to exist.

    Arguments:
        reftime (timestamp):
            The reference time.
        forecast (int):
            The forecast hour.

    Returns:
        str:
            A URL to a GRIB file.
    '''
    reftime = casts.utc_timestamp(reftime).floor('6h')
    now = casts.utc_timestamp('now').floor('6h')
    delta = now - reftime
    if pd.Timedelta(7, 'd') < delta:
        url_fmt = ARCHIVE_URL
    else:
        url_fmt = PROD_URL
    return url_fmt.format(ref=reftime, forecast=forecast)


def grib_path(reftime, forecast):
    '''The path to a GRIB for the given reference and forecast times.

    GRIB forecasts are downloaded to this path and may be deleted once the
    forecast is processed into netCDF. This file does not necessarily exist.

    Arguments:
        reftime (timestamp):
            The reference time.
        forecast (int):
            The forecast hour.

    Returns:
        pathlib.Path:
            The local path for a GRIB file, which may not exist.
    '''
    reftime = casts.utc_timestamp(reftime).floor('6h')
    prefix_fmt = 'nam.{ref.year:04d}{ref.month:02d}{ref.day:02d}'
    filename_fmt = 'nam.t{ref.hour:02d}z.awphys{forecast:02d}.tm00.grib'
    prefix = prefix_fmt.format(forecast=forecast, ref=reftime)
    filename = filename_fmt.format(forecast=forecast, ref=reftime)
    return DATA_DIR / prefix / filename


def nc_path(reftime):
    '''The path to a netCDF for the given reference time.

    NetCDF files are generated after forecasts are processed from the raw
    GRIB data. This file does not necessarily exist.

    Arguments:
        reftime (timestamp):
            The reference time.

    Returns:
        pathlib.Path:
            The local path to a netCDF file, which may not exist.
    '''
    reftime = reftime = casts.utc_timestamp(reftime).floor('6h')
    prefix = f'nam.{reftime.year:04d}{reftime.month:02d}{reftime.day:02d}'
    filename = f'nam.t{reftime.hour:02d}z.awphys.tm00.nc'
    return DATA_DIR / prefix / filename


def _download_grib(reftime, forecast, max_tries=8, timeout=10, fail_fast=False):
    '''Ensure that the GRIB for this reftime and forecast exists locally.

    Arguments:
        reftime (timestamp):
            The reference time to download.
        forecast (int):
            The forecast hour to download
        max_tries (int):
            The maximum number of failed downloads for a single file
            before raising an `IOError`. Exponential backoff is applied
            between attempts, starting at 1 second.
        timeout (int):
            The network timeout in seconds. The government servers are often
            slow to respond.
        fail_fast (bool):
            If true, the download errors are treated as fatal.
            This overrides the `max_tries` argument.

    Returns:
        xarray.Dataset:
            The downloaded dataset.
    '''
    if fail_fast:
        max_tries = 1

    url = grib_url(reftime, forecast)
    path = grib_path(reftime, forecast)
    path.parent.mkdir(exist_ok=True)

    for i in range(max_tries):
        if path.exists():
            break

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

    logger.info(f'reading {path}')
    return xr.open_dataset(path, engine='pynio')


def _process_grib(ds, reftime, forecast):
    '''Process a forecast loaded from GRIB.

    GRIB files contain a forecast for a specific forecast hour at a specific
    reftime, including all NAM data variables for the entire NAM 218 grid.

    This method trims the dataset to the subset of variables and geographic
    region that we are interested in, normalizes variable names and shapes
    to a more consistent format, and adds additional metadata.

    Arguments:
        ds (xarray.Dataset):
            The dataset to process.
        reftime (timestamp):
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
    reftime = casts.utc_timestamp(reftime).floor('6h')
    epoch = casts.utc_timestamp('1970-01-01 00:00')
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

    now = casts.utc_timestamp("now", tz='utc')
    ds.attrs['title'] = 'NAM-UGA, a subset of NAM-NMM for solar forecasting research in Georgia'
    ds.attrs['history'] = f'{now.isoformat()} Initial conversion from GRIB files released by NCEP\n'

    ds = xr.decode_cf(ds)
    return ds


def download(reftime='now', save_nc=True, keep_gribs=False, **kwargs):
    '''Download a forecast for a given reference time.

    The download is skipped for GRIB files in the cache.

    Arguments:
        reftime (timestamp):
            The reference time to open.
        save_nc (bool or None):
            Whether to save the processed forecast in the cache as a netCDF.
        keep_gribs (bool or None):
            Whether to save the raw forecast in the cache as a set of GRIBs.
        max_tries (int):
            The maximum number of failed downloads for a single file
            before raising an `IOError`. Exponential backoff is applied
            between attempts, starting at 1 second.
        timeout (int):
            The network timeout in seconds. The government servers are often
            slow to respond.
        fail_fast (bool):
            If true, the download errors are treated as fatal.
            This overrides the `max_tries` argument.

    Returns:
        xarray.Dataset:
            A dataset for the forecast at this reftime.
    '''
    # We save each GRIB as a netCDF in a temp directory, then reopen all
    # as a single dataset, which we finally persist in the datastore.
    # It is important to persist the intermediate datasets for performance
    # and memory usage.
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        paths = []
        for forecast in FORECAST_PERIOD:
            path = tmpdir / f'{forecast}.nc'
            ds = _download_grib(reftime, forecast)
            ds = _process_grib(ds, reftime, forecast)
            ds.to_netcdf(path)
            paths.append(path)
        ds = _open_mfdataset(paths)

    if save_nc:
        path = nc_path(reftime)
        logger.info(f'writing {path}')
        ds.to_netcdf(path)

    if not keep_gribs:
        for forecast in FORECAST_PERIOD:
            path = grib_path(reftime, forecast)
            logger.info(f'deleting {path}')
            path.unlink()

    return ds


def open(reftimes='now', on_miss='raise', **kwargs):
    '''Open the forecasts for the given reference times.

    Arguments:
        reftimes (timestamp or sequence):
            The reference time(s) to open. The default is to load the most
            recent forecast.
        on_miss (str):
            Determines the behavior on a cache miss:
            - ``'raise'``: Raise a :class:`CacheMiss` exception.
            - ``'download'``: Attempt to download the missing forecast.
            - ``'skip'``: Skip missing forecasts. This mode will raise a
              :class:`CacheMiss` exception only if the resulting dataset
              would be empty.
        **kwargs:
            Additional keyword arguments are forwarded to :func:`download`.

    Returns:
        xarray.Dataset:
            A single dataset containing all forecasts at the given reference
            times.
    '''
    try:
        reftimes = [
            casts.utc_timestamp(reftimes).floor('6h')
        ]
    except TypeError:
        reftimes = [
            casts.utc_timestamp(r).floor('6h')
            for r in reftimes
        ]

    paths = []
    for reftime in reftimes:
        path = nc_path(reftime)
        if path.exists():
            paths.append(path)
        elif on_miss == 'download':
            download(reftime)
            paths.append(path)
        elif on_miss == 'skip':
            continue
        else:
            raise CacheMiss(f'Missing forecast for reftime {reftime}')

    if len(paths) == 0:
        raise CacheMiss('No applicable forecasts were found')

    ds = _open_mfdataset(paths)

    # Reconstruct `time` dimension by combining `reftime` and `forecast`.
    # - `reftime` is the time the forecast was made.
    # - `forecast` is the offset of the data relative to the reftime.
    # - `time` is the time being forecasted.
    ds = ds.assign_coords({'time': ds.reftime + ds.forecast})

    return ds


def open_range(start, stop='now', on_miss='skip', **kwargs):
    '''Open the forecasts for a range of reference times.

    Arguments:
        start (timestamp):
            The first time in the range.
        stop (timestamp):
            The last time in the range.
        on_miss (str):
            Determines the behavior on a cache miss:
            - ``'raise'``: Raise a :class:`CacheMiss` exception.
            - ``'download'``: Attempt to download the forecast.
            - ``'skip'``: Skip missing forecasts.
        **kwargs:
            Additional keyword arguments are forwarded to :func:`download`.

    Returns:
        xarray.Dataset:
            A single dataset containing all forecasts at the given reference
            times.
    '''
    start = casts.utc_timestamp(start).floor('6h')
    stop = casts.utc_timestamp(stop).floor('6h')
    reftimes = pd.date_range(start, stop, freq='6h')
    return open(reftimes, on_miss=on_miss, **kwargs)
