# Data Loaders


## `uga_solar.data.ga_power`

Use this module to access the data feed provided by GA Power. This data is not public.


## `uga_solar.data.nam`

Use this module to access the NAM-NMM weather forecast released by the NCEP, a division of NOAA. This module can access the live feed forecast released by the NCEP as well as an archive that goes back roughly eleven months. This module caches data, so it can be used to build up a sizable dataset over time.

The forecast is released as GRIB files. This module converts those forecasts to a database of netCDF files that can be streamed from disk using `xarray` and `dask`. This enables out-of-core learning.

This is the dataset most will be interested in.
