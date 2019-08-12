Python API Reference
===========================================================================


Modeling Framework
---------------------------------------------------------------------------

Apollo's core API is the modeling framework.

**Model Classes**

.. autosummary::
    :nosignatures:
    :toctree: api

    apollo.models.Model
    apollo.models.IrradianceModel
    apollo.models.NamModel

**Utility Functions**

.. autosummary::
    :nosignatures:
    :toctree: api

    apollo.models.list_models
    apollo.models.list_templates
    apollo.models.load_model
    apollo.models.load_model_at
    apollo.models.load_model_from
    apollo.models.make_estimator
    apollo.models.make_model
    apollo.models.make_model_from


NAM Forecast Data
---------------------------------------------------------------------------

Apollo uses the North American Mesoscale (NAM) forecast system, a numerical weather simulation produced by the National Oceanic and Atmospheric Administration (NOAA). Apollo can be configured to collect NAM forecasts as training data for machine learning models.

**Data Access**

.. autosummary::
    :nosignatures:
    :toctree: api

    apollo.nam.download
    apollo.nam.open
    apollo.nam.open_range
    apollo.nam.CacheMiss

**Geographic Coordinates**

.. autosummary::
    :nosignatures:
    :toctree: api

    apollo.nam.NAM218
    apollo.nam.proj_coords
    apollo.nam.slice_geo

**Useful Constants**

.. autosummary::
    :nosignatures:
    :toctree: api

    apollo.nam.ATHENS_LATLON
    apollo.nam.PLANAR_FEATURES


.. seealso::
    `NAM Home Page <https://www.emc.ncep.noaa.gov/index.php?branch=NAM>`_
        Detailed documentation of NAM.
    `National Centers for Environmental Information <https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/north-american-mesoscale-forecast-system-nam>`_
        Access to raw NAM forecast data.
    `Inventory of File nam.t00z.awphys00.tm00.grib2 <https://www.nco.ncep.noaa.gov/pmb/products/nam/nam.t00z.awphys00.tm00.grib2.shtml>`_
        Catalog of variables included in NAM forecasts. (Apollo does not support every variable.)


Feature Extraction
---------------------------------------------------------------------------

.. autosummary::
    :nosignatures:
    :toctree: api

    apollo.time_of_day
    apollo.time_of_year
    apollo.is_daylight


Time Series Related
---------------------------------------------------------------------------

Timestamps in Apollo adhere to the following conventions:

- Timestamps are always UTC.
- Timezone-naive inputs are interpreted as UTC.
- Timezone-aware inputs in a different timezone are converted to UTC.

Apollo extends common Pandas utilities to support these conventions.

.. autosummary::
    :nosignatures:
    :toctree: api

    apollo.Timestamp
    apollo.DatetimeIndex
    apollo.date_range


Metrics
---------------------------------------------------------------------------

.. autosummary::
    :nosignatures:
    :toctree: api

    apollo.metrics.all
    apollo.metrics.mae
    apollo.metrics.r2
    apollo.metrics.rmse
    apollo.metrics.stdae


Visualizations
---------------------------------------------------------------------------

Apollo includes several visualization routines.

.. autosummary::
    :nosignatures:
    :toctree: api

    apollo.date_heatmap
    apollo.date_heatmap_figure
    apollo.nam_figure


Data Access
---------------------------------------------------------------------------

Apollo stores models and datasets in the *Apollo database*. The database is a regular directory specified by the ``APOLLO_DATA`` environment variable, defaulting to ``/var/lib/apollo``. In the Apollo Docker image, the database is a volume mounted to ``/apollo-data``.

.. autosummary::
    :nosignatures:
    :toctree: api

    apollo.path
