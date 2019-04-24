##################################################
Cheat Sheat
##################################################

.. contents::
    :local:

**************************************************
Starting the Apollo Data Explorer Server
**************************************************

The server can be started by by invoking :mod:`apollo.server.solarserver` with the appropriate arguments. ::

    $ python -m apollo.server.solarserver --host 127.0.0.1 --port 5000 --html "I:\html" --dbdir "I:\db" 
    INFO:  * host:127.0.0.1
    INFO:  * port:5000
    INFO:  * html:I:\html
    INFO:  * html url:/html
    INFO:  * dbdir:I:\db
    INFO:  * dbfile:default.db
    INFO:  * db url:/apollo
    INFO:  * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

The user should specify the IP, port number to listen on, and directories holding
both the databases to access and the static HTML and other files to serve. 

See the documentation on :mod:`apollo.server.solarserver` for more information. 

**************************************************
Creating an empty database for solar farm logs
**************************************************

The database schemas for the solar farm and GAEMN log data 
are stored as SQL scripts in the ``apollo.assets.sql`` directory. An empty database of the correct
format can be created by invoking :mod:`apollo.db.dbinit`. See also 
:func:`apollo.db.converters.init_db`. and  :func:`apollo.db.gaemn.init_db`. ::
 
    $ python -m apollo.db.dbinit c:/test/solar_farm.db
    INFO:  * db:c:/test/solar_farm.db
    INFO: db init...
    INFO: connected...
    INFO: invoking script...
    INFO: db initialized...
    
    $ python -m apollo.db.dbinit c:/test/solar_farm.db
    INFO:  * db:c:/test/solar_farm.db
    INFO: db init...
    INFO: File already exists! Nothing done.
    
    $ python -m apollo.db.dbinit --gaemn c:/test/gaemn1.db
    INFO:  * db:c:/test/gaemn1.db
    INFO: db init...
    INFO: connected...
    INFO: invoking script...
    INFO: db initialized...
    
**************************************************************
Converting csv data for use in a solar farm database.
**************************************************************

The recorded observations files from a GAEMN weather station or the UGA solar 
farm typically must be converted (columns for ``YEAR``, ``MONTH``, ``DAY``, etc., 
must be added, and other transformations are performed.) The conversion can be 
performed using :mod:`apollo.db.convert`. However,  the conversion 
routines can also be invoked as part of the insertion process, 
and so this script need not be invoked directly. 

The below invocations will convert two directories of csv files (with input in slightly 
different formats). See :mod:`apollo.db.converters` for more information. ::

    $ python -m apollo.db.convert --format reapr  --in "C:/test/reapr_in" --out "C:/test/reapr_out"
    INFO:  * format:reapr
    INFO:  * in:C:/test/reapr_in
    INFO:  * out:C:/test/reapr_out

    $ python -m apollo.db.convert --format log  --in "C:/test/gz_in" --out "C:/test/gz_out"
    INFO:  * format:log
    INFO:  * in:C:/test/gz_in
    INFO:  * out:C:/test/gz_out


**************************************************
Inserting csv data into a database
**************************************************

Data in the proper format can be inserted into the solar farm database. Generally, 
entire directories of files are inserted. Users must specify input directory and the database
file to use.  
as well as the original format (``log`` or ``reapr``) of the input files.  ::

    $ python -m apollo.db.insert --format reapr  --in "C:/test/reapr_in" --db "C:/test/solar_farm.db"
    INFO: database insert...
    INFO:  * format:reapr
    INFO:  * in:C:/test/reapr_in
    INFO:  * db:C:/test/solar_farm.db
    INFO:  * table:None
    INFO:  * no convert:False
    INFO:  * no temp:False
    INFO: processing: "C:/test/reapr_in/UGA Solar Tracking Demo IRR 01 01-08 2018.csv
    INFO: Finished: "C:/test/reapr_in/UGA Solar Tracking Demo IRR 01 01-08 2018.csv
    INFO: processing: "C:/test/reapr_in/UGA Solar Tracking Demo IRR 01 09-10 2018.csv
    INFO: Finished: "C:/test/reapr_in/UGA Solar Tracking Demo IRR 01 09-10 2018.csv
    INFO: processing: "C:/test/reapr_in/UGA Solar Tracking Demo IRR 01 11-16 2018.csv
    INFO: Finished: "C:/test/reapr_in/UGA Solar Tracking Demo IRR 01 11-16 2018.csv
    INFO: processing: "C:/test/reapr_in/UGA Solar Tracking Demo IRR 01 17-23 2018.csv
    INFO: Finished: "C:/test/reapr_in/UGA Solar Tracking Demo IRR 01 17-23 2018.csv
    INFO: processing: "C:/test/reapr_in/UGA Solar Tracking Demo IRR 01 24-24 2018.csv
    INFO: Finished: "C:/test/reapr_in/UGA Solar Tracking Demo IRR 01 24-24 2018.csv

    $ python -m apollo.db.insert --format log  --in "C:/test/gz_in" --db "C:/test/solar_farm.db" --table IRRADIANCE
    INFO: database insert...
    INFO:  * format:log
    INFO:  * in:C:/test/gz_in
    INFO:  * db:C:/test/solar_farm.db
    INFO:  * table:IRRADIANCE
    INFO:  * no convert:False
    INFO:  * no temp:False
    INFO: processing: C:/test/gz_in/IRRADIANCE.csv.gz
    INFO: Finished: C:/test/gz_in/IRRADIANCE.csv.gz

See :mod:`apollo.db.converters` for more information.

**************************************************
Creating HTML files from JSON forecasts
**************************************************

Forecasts produced by Apollo are generally saved as JSON. These can be 
converted to a set of HTML pages to be served over the Web. An example of running the 
script for this is given below. The templates used to generate forecast files and an index 
page are stored in ``apollo.assets.templates``. Command line arguments can be 
used instead, however. ::

    $ python -m apollo.server.html -i forecast_json -o html/apollo/forecasts
    INFO:  * in:forecast_json
    INFO:  * out:html/apollo/forecasts
    INFO:  * index:index.html
    INFO:  * template:forecast.html
    
See :mod:`apollo.server.html` for more information.
