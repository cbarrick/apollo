'''A NAM-NMM dataset loader.

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
the data for a single NAM run at a particular reference time,
downloading and preprocessing GRIB files if needed. The actual data
loading logic is encapsulated in the `NAMLoader` class which can be
used for finer grain control over the preprocessing and file system
usage.

The data loading logic works like this:

1. If a netCDF file exists for the dataset, it is loaded immediately
   without any preprocessing.
2. Otherwise any GRIB files required for building the dataset are
   downloaded if they do not already exist.
3. The data is then extracted from the GRIBs. The raw data is subsetted
   to an area encompasing Georgia, and only a subset of the features
   are extracted. The elevation and time axes are reconstructed from
   multiple GRIB features.
4. The dataset is then saved to a netCDF file, and the GRIB files are
   removed.

The dataset is returned as an `xarray.Dataset`, and each variable has
exactly five dimensions: ref, time, x, y, and z. The z-axis for each
variable has a different name depending on the type of index measuring
the axis, e.g. `heightAboveGround` for height above the surface in
meters or `isobaricInhPa` for isobaric layers. the ref- and time-axes
coresponds to the reference and forecast times respectively. The names
of the variables follow the pattern `FEATURE_LAYER` where `FEATURE` is
a short identifier for the feature being measured and `LAYER` is the
type of z-axis used by the variable, e.g. `t_isobaricInhPa` for the
temperature at the isobaric layers.

A mapping of all feature identifiers to descriptions is given by the
module-level constant `FEATURES`.
'''

from datetime import datetime, timedelta, timezone
from itertools import groupby
from logging import getLogger
from pathlib import Path
from time import sleep

import numpy as np
import scipy as sp
import scipy.spatial.distance
import pygrib
import requests
import xarray as xr

logger = getLogger(__name__)

PROD_URL = 'http://nomads.ncep.noaa.gov/pub/data/nccf/com/nam/prod/nam.{ref.year:04d}{ref.month:02d}{ref.day:02d}/nam.t{ref.hour:02d}z.awphys{forecast:02d}.tm00.grib2'
ARCHIVE_URL = 'https://nomads.ncdc.noaa.gov/data/meso-eta-hi/{ref.year:04d}{ref.month:02d}/{ref.year:04d}{ref.month:02d}{ref.day:02d}/nam_218_{ref.year:04d}{ref.month:02d}{ref.day:02d}_{ref.hour:02d}00_{forecast:03d}.grb'

LOCAL_GRIB_FMT = 'nam.{ref.year:04d}{ref.month:02d}{ref.day:02d}/nam.t{ref.hour:02d}z.awphys{forecast:02d}.tm00.grib2'
LOCAL_CDF_FMT = 'nam.{ref.year:04d}{ref.month:02d}{ref.day:02d}/nam.t{ref.hour:02d}z.awphys.tm00.nc'

DATETIME_FMT = '%Y%m%d %H%M'

FORECAST_INDEX = tuple(range(0, 36)) + tuple(range(36, 85, 3))

DEFAULT_FEATURES = ['pres', 'vis', 'tcc', 't', 'r', 'u', 'v', 'w']

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


def load(ref_time=None, data_dir='.', url_fmt=None, save_netcdf=True, keep_gribs=False):
    '''Load a NAM-NMM dataset for the given reference time.

    Args:
        ref_time (datetime or str):
            The reference time of the data set. It is rounded down to the
            previous model run. It may be given as a string with the format
            '%Y%m%d %H%M'. The default is the current time.
        data_dir (path-like):
            The base path for the dataset.
        url_fmt (string):
            The format for GRIB URLs. It uses the keys `ref` and `forecast`
            for the reference time and forecast hour respectively. The
            default is either the production URL from NCEP or the archive
            URL from NCDC depending on the reference time.
        save_netcdf (bool):
            If true, save the dataset to a netCDF file.
            This argument is ignored if loading from netCDF.
        keep_gribs (bool):
            If true, the GRIB files are not deleted.
            This argument is ignored if loading from netCDF.
    '''
    loader = NAMLoader(ref_time=ref_time, data_dir=data_dir, url_fmt=url_fmt)
    return loader.load(save_netcdf=save_netcdf, keep_gribs=keep_gribs)


def map_slice(lats, lons, center=(32.8, -83.6), apo=40):
    '''Compute a slice of a map projection.

    Defaults to a square centered on Macon, GA that encompases all of
    Georgia, Alabama, and South Carolina.

    The slice object returned can be used to subset a projected grid. E.g:

        data, lats, lons = get_data_from_foo()
        subset = map_slice(lats, lons)
        data, lats, lons = data[subset], lats[subset], lons[subset]

    Args:
        lats (2d array):
            The latitudes for each cell of the projection.
        lons (2d array):
            The longitudes for each cell of the projection.
        center (pair of float):
            The center of the slice as a `(latitude, longitude)` pair.
        apo (int):
            The apothem of the subset in grid units.
            I.e. the distance from the center to the edge.

    Returns:
        A pair of slices characterizing the subset.
    '''
    latlons = np.stack((lats.flatten(), lons.flatten()), axis=-1)
    target = np.array([center])
    dist = sp.spatial.distance.cdist(target, latlons)
    am = np.argmin(dist)
    i, j = np.unravel_index(am, lats.shape)
    return slice(i - apo, i + apo + 1), slice(j - apo, j + apo + 1)


class NAMLoader:
    '''A data loader for the NAM-NMM dataset.

    The main method is `load` which is equivalent to the module-level function
    of the same name with the additional functionality of controling the
    feature list and grid subset in the preprocessing step.

    The `load` method is built from a pipeline of the `download`, `unpack`, and
    `repack` methods. `download` ensures that the GRIB files exist locally,
    `unpack` extracts a subset of the data, and `repack` recombines the data
    into an `xarray.Dataset` which can be serialized to a netCDF file. Custom
    preprocessing pipelines can be built from these methods.

    Additionally, this class includes helpers to extract lat-lon grids from the
    GRIB files, compute grid subsets, and access the files of the dataset.
    '''

    def __init__(self, ref_time=None, data_dir='.', url_fmt=None):
        '''Creates a NAM data loader for the given reference time.

        Args:
            ref_time (datetime):
                The reference time of the data set. It is rounded down to the
                previous model run. It may be given as a string with the format
                '%Y%m%d %H%M'. The default is the current time.
            data_dir (Path):
                The base path for the dataset.
            url_fmt (string):
                The format for GRIB URLs. It uses the keys `ref` and `forecast`
                for the reference time and forecast hour respectively. The
                default is either the production URL from NCEP or the archive
                URL from NCDC depending on the reference time.
        '''
        now = datetime.now(timezone.utc)

        # The reference time must be in UTC
        if not ref_time:
            ref_time = now
        elif isinstance(ref_time, str):
            ref_time = datetime.strptime(ref_time, DATETIME_FMT).astimezone(timezone.utc)
        else:
            ref_time = ref_time.astimezone(timezone.utc)

        # The reference time is rounded to the previous 0h, 6h, 12h, or 18h
        ref_time = ref_time.replace(
            hour=(ref_time.hour // 6) * 6,
            minute=0,
            second=0,
            microsecond=0,
        )

        # The default url_fmt is based on the reference time.
        if not url_fmt:
            days_delta = (now - ref_time).days
            if days_delta > 7:
                url_fmt = ARCHIVE_URL
            else:
                url_fmt = PROD_URL

        self.ref_time = ref_time
        self.data_dir = Path(data_dir)
        self.url_fmt = url_fmt

    def load(self, features=None, subset=None, save_netcdf=True, keep_gribs=False):
        '''Load the dataset, downloading and preprocessing GRIBs as necessary.

        If the dataset exists as a local netCDF file, it is loaded and
        returned. Otherwise, any missing GRIB files are downloaded and
        preprocessed into an xarray Dataset. The dataset is then saved as a
        netCDF file, the GRIBs are deleted, and the dataset is returned.

        Args:
            features (list of str):
                The list of feature IDs to extract from the GRIBs.
                This argument is ignored if loading from netCDF.
                The defaults are pressure, visibility, cloud coverage,
                temperature, relative humidity, and the three wind vectors.
            subset (dict):
                The grid subset to extract from the GRIBs, as a dict of keyword
                arguments to pass to `NAMLoader.map_slice`. This argument is
                ignored if loading from netCDF. The default is an area
                encompasing all of Georgia.
            save_netcdf (bool):
                If true, save the dataset to a netCDF file.
                This argument is ignored if loading from netCDF.
            keep_gribs (bool):
                If true, the GRIB files are not deleted.
                This argument is ignored if loading from netCDF.
        '''
        # If the dataset already exists, just load it.
        if self.local_cdf.exists():
            return xr.open_dataset(str(self.local_cdf))

        # Otherwise download and preprocess the gribs.
        self.download()

        # The subset is passed in as a dict.
        # Convert it to slices or get the default.
        if not subset:
            subset = self.map_slice()
        else:
            subset = self.map_slice(**subset)

        # Load the dataset from the gribs.
        if features is None:
            features = DEFAULT_FEATURES
        data = self.unpack(features, subset)
        data = self.repack(data)

        # Save as netCDF.
        if save_netcdf:
            data.to_netcdf(str(self.local_cdf))

        # Delete the gribs.
        if not keep_gribs:
            for grib_path in self.local_gribs:
                grib_path.unlink()

        return data

    def download(self, force=False):
        '''Download the missing GRIB files for this dataset.

        Args:
            force (bool):
                Download the GRIB files even if they already exists locally.
        '''
        for path, url in zip(self.local_gribs, self.remote_gribs):
            path.parent.mkdir(exist_ok=True)

            # No need to download if we already have the file.
            # TODO: Can we check that the file is valid before skipping it?
            if path.exists() and not force:
                continue

            # Attempt download.
            # In case of error, retry with exponential backoff.
            # Give up after 10 tries (~35 minutes).
            max_tries = 10
            timeout = 10 # the servers are kinda slow
            for i in range(max_tries):
                try:
                    with path.open('wb') as fd:
                        logger.info('Downloading {}'.format(url))
                        r = requests.get(url, timeout=timeout, stream=True)
                        r.raise_for_status()
                        for chunk in r.iter_content(chunk_size=128):
                            fd.write(chunk)
                    break
                except IOError as err:
                    # IOError includes both system and HTTP errors.
                    path.unlink()
                    logger.info(err)
                    if i != max_tries - 1:
                        delay = 2 ** i
                        logger.info('Download failed, retrying in {}s'.format(delay))
                        sleep(delay)
                    else:
                        logger.error('Download failed, giving up')
                        raise err
                except (Exception, SystemExit, KeyboardInterrupt) as err:
                    # Delete partial file in case of keyboard interupt etc.
                    path.unlink()
                    raise err

    def unpack(self, features, subset):
        '''Unpacks and subsets the local GRIB files.

        Args:
            features (list of str):
                The short names of features to include in the dataset.
            subset (2-tuple of slice):
                The subset to extract in the form `(x, y)` where `x` and `y`
                are slices for their respective dimensions.

        Returns:
            A list of `DataArray`s for each feature in the GRIB files.
        '''
        variables = []
        for path in self.local_gribs:
            logger.info('Processing {}'.format(path))
            grbs = pygrib.open(str(path))
            for g in grbs.select(shortName=features):

                layer_type = g.typeOfLevel
                if layer_type == 'unknown':
                    layer_type = 'z' + str(g.typeOfFirstFixedSurface)

                name = '_'.join([g.shortName, layer_type])

                ref_time = datetime(g.year, g.month, g.day, g.hour, g.minute, g.second)
                ref_time = np.datetime64(ref_time)

                forecast = timedelta(hours=g.forecastTime)
                forecast = np.timedelta64(forecast)
                forecast = ref_time + forecast

                lats, lons = g.latlons()                   # lats and lons are in (y, x) order
                lats, lons = lats.T, lons.T                # transpose to (x, y)
                lats, lons = lats[subset], lons[subset]    # subset applies to (x, y) order
                lats, lons = np.copy(lats), np.copy(lons)  # release reference to the grib

                values = g.values.T                 # g.values is (y, x), transpose to (x, y)
                values = values[subset]             # subset applies to (x, y) order
                values = np.copy(values)            # release reference to the grib
                values = np.expand_dims(values, 0)  # add reference time axis: (ref, x, y)
                values = np.expand_dims(values, 1)  # add forecast axis: (ref, time, x, y)
                values = np.expand_dims(values, 4)  # add z axis: (ref, time, x, y, z)

                dims = ['ref', 'time', 'x', 'y', 'z']

                coords = {
                    'ref': ('ref', [ref_time], {
                        'long_name': 'reference time',
                        'standard_name': 'forecast_reference_time',
                    }),
                    'time': ('time', [forecast], {
                        'long_name': 'validity time',
                        'standard_name': 'time',
                        'axis': 'T',
                    }),
                    'x': ('x', np.arange(values.shape[2]), {
                        'coordinates': 'lat lon',
                        'axis': 'X',
                    }),
                    'y': ('y', np.arange(values.shape[3]), {
                        'coordinates': 'lat lon',
                        'axis': 'Y',
                    }),
                    'z': ('z', [g.level], {
                        'units': g.unitsOfFirstFixedSurface,
                        'axis': 'Z',
                    }),
                    'lat': (('x', 'y'), lats, {
                        'long_name': 'latitude',
                        'standard_name': 'latitude',
                        'units': 'degrees_north',
                    }),
                    'lon': (('x', 'y'), lons, {
                        'long_name': 'longitude',
                        'standard_name': 'longitude',
                        'units': 'degrees_east',
                    }),
                }

                attrs = {
                    'layer_type': layer_type,
                    'short_name': g.shortName,
                    'standard_name': g.cfName or g.name.replace(' ', '_').lower(),
                    'units': g.units,
                }

                arr = xr.DataArray(name=name, data=values, dims=dims, coords=coords, attrs=attrs)
                variables.append(arr)

        return variables

    def repack(self, variables):
        '''Packs a list of `DataArray` features into a `Dataset`.

        Variables of the same name are concatenated along the z- and time-axes.
        If a feature has the attribute `layer_type`, then its value becomes the
        new name of the z-axis. This allows features with different kinds of
        z-axes to live in the same dataset.

        Args:
            variables (list):
                The features to recombine.

        Returns:
            The reconstructed `Dataset`.
        '''
        logger.info('Sorting variables')
        variables = sorted(variables, key=lambda v: v.time.data[0])
        variables = sorted(variables, key=lambda v: v.name)

        logger.info('Reconstructing the z dimensions')
        tmp = []
        for k, g in groupby(variables, lambda v: (v.name, v.time.data[0])):
            v = xr.concat(g, dim='z')
            v = v.rename({'z': v.attrs['layer_type']})
            tmp.append(v)
        variables = tmp

        logger.info('Reconstructing the time dimension')
        tmp = []
        for k, g in groupby(variables, lambda v: (v.name)):
            v = xr.concat(g, dim='time')
            tmp.append(v)
        variables = tmp

        logger.info('Collecting the dataset')
        dataset = xr.merge(variables)
        return dataset

    def map_slice(self, center=(32.8, -83.6), apo=40):
        '''Compute a slice of a map projection.

        Defaults to a square centered on Macon, GA that encompases all of
        Georgia, Alabama, and South Carolina.

        The slice object returned can be used to subset a projected grid. E.g:

            data, lats, lons = get_data_from_foo()
            subset = loader.map_slice()
            data, lats, lons = data[subset], lats[subset], lons[subset]

        Note that the slice applies to data in `(x, y)` order, but the GRIB
        data is in `(y, x)` order by default.

        Args:
            center (pair of float):
                The center of the slice as a `(latitude, longitude)` pair.
            apo (int):
                The apothem of the subset in grid units.
                I.e. the distance from the center to the edge.

        Returns:
            A pair of slices characterizing the subset.
        '''
        # Get lats and lons from the first variable of the first grib.
        paths = tuple(self.local_gribs)
        first_file = str(paths[0])
        grbs = pygrib.open(first_file)
        g = grbs[1]  # indices start at 1
        lats, lons = g.latlons()
        lats, lons = lats.T, lons.T  # transpose to (x, y)
        return map_slice(lats, lons, center, apo)

    @property
    def remote_gribs(self):
        '''An iterator over the URLs for the GRIB files in this dataset.'''
        for i in FORECAST_INDEX:
            yield self.url_fmt.format(ref=self.ref_time, forecast=i)

    @property
    def local_gribs(self):
        '''An iterator over the paths to the local GRIB files in this dataset.'''
        for i in FORECAST_INDEX:
            yield self.data_dir / Path(LOCAL_GRIB_FMT.format(ref=self.ref_time, forecast=i))

    @property
    def local_cdf(self):
        '''The path to the local netCDF file for this dataset.'''
        return self.data_dir / Path(LOCAL_CDF_FMT.format(ref=self.ref_time))


if __name__ == '__main__':
    import argparse
    import logging

    parser = argparse.ArgumentParser(description='Download and preprocess the NAM-NMM dataset.')
    parser.add_argument('--log', type=str, help='Set the log level')
    parser.add_argument('--ref', type=lambda x: datetime.strptime(x, '%Y-%m-%dT%H00'), help='Set the reference time')
    parser.add_argument('dir', nargs='?', default='.', type=str, help='Base directory for downloads')
    args = parser.parse_args()

    log_level = args.log or 'INFO'
    logging.basicConfig(level=log_level, format='{message}', style='{')

    now = datetime.now(timezone.utc)
    ref_time = args.ref.astimezone(timezone.utc)
    one_hour = timedelta(hours=1)

    data_dir = args.dir

    print(ref_time, now, ref_time < now)

    while ref_time < now:
        load(ref_time, data_dir=data_dir)
