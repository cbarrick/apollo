# -*- coding: utf-8 -*-
"""Creates an SQLite database from a set of GAEMN 15-minute observation files. 

It is assumed that each observation file consists of local weather observations, 15-minutes apart,
from a single station (a site). It's also assumed that each row contains the following 43 numerical fields. 

.. code-block:: python

    ["SITEID", "YEAR", "JULIANDAY", "TIMEOFDAY", "JULIANDAYHOUR", 
    "AIRTEMPERATURE", "HUMIDITY", "DEWPOINT", "VAPORPRESSURE",
    "VAPORPRESSUREDEFICIT", "BAROMETRICPRESSURE", "WINDSPEED", "WINDDIRECTION",
    "STANDARDDEVIATION", "MAXIMUMWINDSPEED", "TIMEOFMAXIMUMWINDSPEED",
    "SOILTEMPERATURE2CM", "SOILTEMPERATURE5CM", "SOILTEMPERATURE10CM",
    "SOILTEMPERATURE20CM", "SOILTEMPERATUREA", "SOILTEMPERATUREB",
    "SOILMOISTURE", "PAN", "EVAP", "WATERTEMPERATURE", "SOLARRADIATION",
    "TOTALSOLARRADIATION", "PAR", "TOTALPAREINSTEIN", "NETRADIATION",
    "TOTALNETRADIATION", "RAINFALL", "RAINFALL2", "MAXRAINFALL",
    "TIMEOFMAXRAINFALL", "MAXRAINFALL2", "TIMEOFMAXRAINFALL2", "LEAFWETNESS",
    "WETNESSFREQUENCY", "BATTERYVOLTAGE", "FUELTEMPERATURE", "FUELMOISTURE"]

Date and time information is split across multiple fields 
(``YEAR``, ``JULIANDAY``, ``TIMEOFDAY``). To insert the data into a database, the 
following additional fields are created: ``TIMESTAMP`` (number of milliseconds since mindnight January 
1st 1970), ``MONTH`` (1-12), ``DAY`` (1-31), ``HOUR`` (0-23), ``MINUTE`` (0-59), and ``DAYOFYEAR`` (1-366). 
Also, the following preexisting fields are dropped: ``JULIANDAY`` and ``TIMEOFDAY``. The final format is thus: 
    
.. code-block:: python

    ["SITEID", "YEAR", "MONTH", "DAY", "HOUR", "MINUTE", "DAYOFYEAR","JULIANDAYHOUR", 
    "AIRTEMPERATURE", "HUMIDITY", "DEWPOINT", "VAPORPRESSURE",
    "VAPORPRESSUREDEFICIT", "BAROMETRICPRESSURE", "WINDSPEED", "WINDDIRECTION",
    "STANDARDDEVIATION", "MAXIMUMWINDSPEED", "TIMEOFMAXIMUMWINDSPEED",
    "SOILTEMPERATURE2CM", "SOILTEMPERATURE5CM", "SOILTEMPERATURE10CM",
    "SOILTEMPERATURE20CM", "SOILTEMPERATUREA", "SOILTEMPERATUREB",
    "SOILMOISTURE", "PAN", "EVAP", "WATERTEMPERATURE", "SOLARRADIATION",
    "TOTALSOLARRADIATION", "PAR", "TOTALPAREINSTEIN", "NETRADIATION",
    "TOTALNETRADIATION", "RAINFALL", "RAINFALL2", "MAXRAINFALL",
    "TIMEOFMAXRAINFALL", "MAXRAINFALL2", "TIMEOFMAXRAINFALL2", "LEAFWETNESS",
    "WETNESSFREQUENCY", "BATTERYVOLTAGE", "FUELTEMPERATURE", "FUELMOISTURE"]

The input file is read using :func:`pandas.read_csv`. It is assumed that the 
first row contains the header strings described above. 

As an intervediate step, the input is converted and then saved to a seaparate csv file.
A database having the same name as the site is created, and the converted data 
is inserted into the ``OBS`` table of the database. The SQL script for creating
the database is stored in the source code of this script.

Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:: 
    
    $ python -m apollo.db.gaemn --help
    usage: gaemn.py [-h] -i in -o out [--noconvert] [--csv]
    
    $ python -m apollo.db.gaemn --in "c:/gaemn/in/" --out "c:/gaemn/out/" 
    INFO:  * in:C:/gaemn/in/
    INFO:  * out:C:/gaemn/out/
    INFO: ...in: c:/gaemn/in/ALAPAHA.csv
    INFO: ...out: c:/gaemn/out/ALAPAHA.db
    INFO: ...in: c:/gaemn/in/DUNWOODY.csv
    INFO: ...out: c:/gaemn/out/DUNWOODY.db
    INFO: ...in: c:/gaemn/in/GRIFFIN.csv    
    INFO: ...out: c:/gaemn/out/GRIFFIN.db

    $ python -m apollo.db.gaemn --in "c:/gaemn/in/" --out "c:/gaemn/out/" --csv
    INFO:  * in:C:/gaemn/in/
    INFO:  * out:C:/gaemn/out/
    INFO: ...in: c:/gaemn/in/ALAPAHA.csv
    INFO: ...out: c:/gaemn/out/ALAPAHA.out.csv.gz
    INFO: ...in: c:/gaemn/in/DUNWOODY.csv
    INFO: ...out: c:/gaemn/out/DUNWOODY.out.csv.gz
    INFO: ...in: c:/gaemn/in/GRIFFIN.csv    
    INFO: ...out: c:/gaemn/out/GRIFFIN.out.csv.gz

"""

import sqlite3
import argparse
import os
import pandas as pd
import numpy as np
import datetime
import traceback
import logging
import apollo.db.dbapi as dbapi 
import apollo.assets.api as assets
from pathlib import Path


logger = logging.getLogger(__name__)

# GAEMN create table script, lazily read from the assets folder at runtime. 
_SQL = None
# path to gaemn sql. 
_GAMEN_SQL = 'assets/sql/gaemn.sql'

# The SQL script for creating the database. Observations will be stored 
# in a table called OBS
def _get_sql():
    """Read the GAEMN SQL create table script from the assets folder
    """
    global _SQL
    if _SQL == None:
        _SQL = assets.get_asset_string(_GAMEN_SQL)
    return _SQL


# The list of GAEMN sites. Each will be processed. 
SITES = ["ALAPAHA",
        "ALBANY",
        "ALBANYMC",
        "ALMA",
        "ALPHARET",
        "ARABI",
        "ARLINGT",
        "ATLANTA",
        "ATTAPUL",
        "BAXLEY",
        "BLAIRSVI",
        "BLEDSOE",
        "BLURIDGE",
        "BOWEN",
        "BRUNSW",
        "BYROMVIL",
        "BYRON",
        "CAIRO",
        "CALHOUN",
        "CALLAWAY",
        "CAMILLA",
        "CLARKSHI",
        "CORDELE",
        "COVING",
        "CUMMINGS",
        "DAHLON",
        "DALLAS",
        "DANVILLE",
        "DAWSON",
        "DEARING",
        "DEMPSEY",
        "DIXIE",
        "DONALSON",
        "DOUGLAS",
        "DUBLIN",
        "DUCKER",
        "DULUTH",
        "DUNWOODY",
        "EATONTON",
        "ELBERTON",
        "ELLIJAY",
        "FLOYD",
        "FTVALLEY",
        "GAINES",
        "GEORGETO",
        "GRIFFIN",
        "HATLEY",
        "HHERC",
        "HOMERV",
        "HOWARD",
        "JONESB",
        "JVILLE",
        "LAFAYET",
        "MCRAE",
        "MIDVILLE",
        "MOULTRIE",
        "NAHUNTA",
        "NEWTON",
        "OAKWOOD",
        "ODUM",
        "OSSABAW",
        "PLAINS",
        "SANLUIS",
        "SASSER",
        "SAVANNAH",
        "SHELLMAN",
        "SKIDAWAY",
        "SNEADS",
        "SPARTA",
        "STATES",
        "TENNILLE",
        "TIFTON",
        "TIGER",
        "TYTY",
        "UNADILLA",
        "VALDOSTA",
        "VIDALIA",
        "VIENNA",
        "WANSLEY",
        "WATHORT",
        "WATUGA",
        "WATUSDA",
        "WOODBINE"]

# The input format of the GAEMN 15 Minute log files. 
# There should be one CSV file for each site. 
# This list is not used here; its included just for the sake of completeness. 
HEADERS = [ "SITEID",
            "YEAR",
            "JULIANDAY",
            "TIMEOFDAY",
            "JULIANDAYHOUR",
            "AIRTEMPERATURE",
            "HUMIDITY",
            "DEWPOINT",
            "VAPORPRESSURE",
            "VAPORPRESSUREDEFICIT",
            "BAROMETRICPRESSURE",
            "WINDSPEED",
            "WINDDIRECTION",
            "STANDARDDEVIATION",
            "MAXIMUMWINDSPEED",
            "TIMEOFMAXIMUMWINDSPEED",
            "SOILTEMPERATURE2CM",
            "SOILTEMPERATURE5CM",
            "SOILTEMPERATURE10CM",
            "SOILTEMPERATURE20CM",
            "SOILTEMPERATUREA",
            "SOILTEMPERATUREB",
            "SOILMOISTURE",
            "PAN",
            "EVAP",
            "WATERTEMPERATURE",
            "SOLARRADIATION",
            "TOTALSOLARRADIATION",
            "PAR",
            "TOTALPAREINSTEIN",
            "NETRADIATION",
            "TOTALNETRADIATION",
            "RAINFALL",
            "RAINFALL2",
            "MAXRAINFALL",
            "TIMEOFMAXRAINFALL",
            "MAXRAINFALL2",
            "TIMEOFMAXRAINFALL2",
            "LEAFWETNESS",
            "WETNESSFREQUENCY",
            "BATTERYVOLTAGE",
            "FUELTEMPERATURE",
            "FUELMOISTURE"]


def init_db(db_file):
    """Initialize the database, using the given database file. 
    
    This will create a SQLite database file for log data 
    using a hard coded schema (a set of SQL ``CREATE TABLE`` statements is stored here.)
    
    After the database has been created, log data can be inserted into it. 

    If the database file to create already exists, nothing is done.

    Arguments:
        db_file (str): The database file to create. 

    Returns: 
        str:
            The SQL statements used to construct the database schema. 
    """
    logging.info("db init...")
    path = Path(db_file)
    if(path.exists()):
        return False
    db_file = str(db_file)
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        logging.info("connected...")
        logging.info("invoking script...")
        conn.executescript(_get_sql())
        logging.info("db initialized...")
    except Exception as e:
        logging.error("error creating db.\n" + str(e))
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass
    return True




  
def convert_df(df):
    """Process a data frame created from a GAEMN observation file. 
    
    This is intended to be used immediately after the observation file has been 
    read and converted into a :class:`pandas.DataFrame`.
    
    * The following fields are added: ``TIMESTAMP``, ``MONTH``, ``DAY``, ``HOUR``, ``MINUTE``, and ``DAYOFYEAR``. 
    * The following fields are dropped: ``JULIANDAY`` and ``TIMEOFDAY``. 
    
    Arguments:
        df (pandas.DataFrame): The dataframe to process. 
    """
    try:
        dates_temps = df.apply(_extract_gaemn_date, axis=1)
        dates = pd.DatetimeIndex(dates_temps)
        # add more date-time fields, for faster query execution
        df.insert(2, "DAYOFYEAR",   dates.dayofyear)
        df.insert(2, "MINUTE", dates.minute)
        df.insert(2, "HOUR", dates.hour)
        df.insert(2, "DAY", dates.day) 
        df.insert(2, "MONTH", dates.month) 
        df.insert(1, "TIMESTAMP", dates_temps.astype(np.int64)/int(1e9))
        df.drop(columns=['JULIANDAY', 'TIMEOFDAY'] ,  axis=1, inplace=True)
    except Exception as e:
        logger.error(f"Error converting dataframe. {e}")
        traceback.print_exc()

    
def csv_to_df(filename, convert=True):
    """Read in a csv or gzipped csv file, returning it as a :class:`pandas.DataFrame`
    instance. 
    
    The entire input file is read into memory as a :class:`pandas.DataFrame`. 
    If ``convert==True``, then the following modifications are made to the :class:`pandas.DataFrame`: 
    
    * The following fields are added: ``TIMESTAMP``, ``MONTH``, ``DAY``, ``HOUR``, ``MINUTE``, and ``DAYOFYEAR``. 
    * The following are dropped: ``JULIANDAY`` and ``TIMEOFDAY``. 
    
    Arguments:
        filename (str): The file to process.
        convert (boolean): Inddicates whether the data should be converted.
    Returns:
        pandas.DataFrame: 
            The generated dataframe. 
    """
    df = None
    try:
        df = pd.read_csv(filename)
        if not convert:
            return df
        convert_df(df)
    except Exception as e:
        logger.error(f"Error converting {filename} to dataframe. {e}")
        traceback.print_exc()
    return df


def create_db(df, dbfile):
    """Insert a :class:`pandas.DataFrame` into a database file. 
    
    It is assumed that the :class:`pandas.DataFrame` instance is in the correct
    format for the database. 
    
    Arguments:
        df (pandas.DataFrame): The dataframe to insert. 
        dbfile (str): The database file to create. 
    """    
    try:
        handler = dbapi.DBHandler(str(dbfile))
        handler.connect()
        handler.executescript(_get_sql(), commit=True)
        handler.insert_dataframe(df, "OBS")
        handler.close()
    except Exception as e: 
        logger.error(f"Error inserting dataframe into {dbfile}. {e}")
        traceback.print_exc()

def _extract_gaemn_date(row):
    """Extract a datetime object from a GAEMN 15 Min observation.
    
    Since timestamp information is spread across multiple columns
    (``YEAR``, ``JULIANDAY``, ``TIMEOFDAY``), the  datetime object must be 
    constructed by examining each of them. 
    
    Arguments:
        row: A row of data from a log GAEMN log file. 
    Returns:
        datetime.datetime:
            A timestamp constructed from the (``YEAR``, ``JULIANDAY``, and ``TIMEOFDAY``.
    """
    tod = _get_hour(row['TIMEOFDAY'])
    return datetime.datetime(_to_int(row['YEAR']),1,1,tzinfo=datetime.timezone(datetime.timedelta(0))) + datetime.timedelta(days=(tod['day'] + _to_int(row['JULIANDAY'])-1), hours=tod['hour'], minutes=tod['minute'])
        
def _to_int(i):
    """Converts string representations of numbers  to ints, via float.
    
    The default int parser will not work for '0.0', for instance, and so we 
    parse the string as a float first. 
    
    Arguments:
        i (str): A string representation of an integer-like number. 
    Returns:
        int: 
            The value cast as an int.
    """
    return int(float(i))

def _get_hour(h):
    """Extract the hour, minutes, and day (0 or 1) from time strings like '115' 
    (01:15) and '1345' (13:45). For '2400', we need to convert to 00:00 and 
    add 1 to the day. The result in that case would be:
        
    >>> {'hour':0, 'minute':0, 'day':1}. 

    Arguments:
        h (str): A string representation of an hour and minutes.
    Returns:
        dict: 
            A diction of the form ``{'hour':H, 'minute':M, and 'day':D}``
    """
    if h >= 2400:
        return {'hour':0,'minute':0,'day':1}
    return {'hour':h//100,'minute':h%100,'day':0}
    

def site_summary_strings():
    """Generate a dictionary describing the gaemn data sources. Used in ``sources.json``.

    The ``sources.json`` stores a list of databases and their locations for use by the 
    Apollo web server. 
    
    The results of this function will have the form shown below. ``id`` is a string identifier for the 
    site. ``label`` is a human-readable name for the site. The two dates are used 
    in the web interface to configure widgets for picking dates. The schema indicates a 
    file that holds information on the structure of the database. See :mod:`apollo.server.schemas` for more information. 
    
    .. code-block:: python
    
        {
        "id":"GRIFFIN"
        "label":"Griffin"
        "ext":".db"
        "schema":"gaemn15min"
        "initial_start":"2013-01-01"
        "initial_stop":"2013-01-02"
        }
    """
    for site in SITES:
        print("{"+f'"id":"{site}", "label":"{site.title()}", "schema":"gaemn15min",   "initial_start": "2013-01-01", "initial_stop": "2013-01-02"'+"},")
    


def create_gaemn_dbs(indir, outdir,  convert=True, usedb=True, compression = 'gzip'):
    """Process a directory of GAEMN log files, converting each file and 
    creating a corresponding SQLite database or csv file for it. 
    
    Input files are assumed to have the name SITENAME.csv or SITENAME.csv.gz. 
    These should be stored in the  path indicated by 'indir'. 
    
    The input files can be 'converted' first, which adds additional TIMESTAMP information.
    
    It if possible to save the results in either a SQLite database or (possibly compressed) csv file. 
    A static SQL create table script is used to create the database file. 

    Arguments:
        indir (str): The directory holding observation files to process.
        outdir (str): The directory to hold the generated output files. 
        convert (bool): Indicates whether input files should be converted first. 
        usedb (bool): Indicates whether the converted data should be converted to a databse or else saved to csv. 
        compression (str): Use compression if saving to csv. Default is 'gzip'.
    """
    for root, dirs, files in os.walk(indir):
        for filename in files:
            logger.debug(filename)
            try:
                infile = Path(root)/filename
                logger.info(f"...in: {infile}")
                site = Path(filename).parts[-1].split('.')[0]
                df = csv_to_df(str(infile), convert=convert)
                if not isinstance(df,pd.DataFrame):
                    continue
                if usedb:
                    dbfile = Path(outdir)/(site + ".db")
                    if dbfile.exists():
                        os.remove(dbfile)
                    create_db(df, dbfile)
                    logger.info(f"...out: {dbfile}")
                else:
                    zipped = site + ".out.csv.gz"
                    outfile = Path(outdir)/zipped
                    df.to_csv(outfile, index = None, header=True, compression=compression)
                    logger.info(f"...out: {outfile}")
            except Exception as e:
                logger.error(f"Error inserting into database. {e}")
            
def config_from_args():
    parser = argparse.ArgumentParser(description="Utility function for creating database files from GAEMN 15-minute observation files. "\
                                     +"You should specify the directory of input csv files as well as the output directory. ")
                                     
    parser.add_argument('-i', '--in', metavar='in', type=str, dest='indir',default=None,required=True,help='the directory to convert.')
    parser.add_argument('-o', '--out', metavar='out', type=str, dest='outdir',default=None,required=True,help='the directory in which to write output.')
    parser.add_argument('--noconvert',  action='store_true',help='do not convert the input file.')
    parser.add_argument('--csv',  action='store_true',help='store the results in a csv file rather than an SQLite database.')

    args = parser.parse_args()

    logging.basicConfig(format='[{asctime}] {levelname}: {message}', style='{', level="INFO")
    logging.info(" * in:"+str(args.indir))
    logging.info(" * out:"+str(args.outdir))
    
    return args


if __name__ == "__main__":
    args = config_from_args()
    convert= True
    usedb= True
    if args.noconvert:
        convert= False
    if args.csv:
        usedb= False
    if True:
        create_gaemn_dbs(args.indir,args.outdir, convert=convert, usedb=usedb)
