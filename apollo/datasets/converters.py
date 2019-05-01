# -*- coding: utf-8 -*-
"""Data conversion and database insertion routines for UGA solar farm log data.
 
These routines are specifically designed for data in one of two formats, both
encoding log data taken from the UGA solar farm. Data in the first format 
was originally stored in many small, gzipped .log files that were pushed to UGA servers. 
Data in the second format was downloaded manually from  REAPR (Research and 
Environmental Affairs Project Reporting), an internal Southern  Company website.
 
The data is for 8 "modules", and for each of these a database table exists 
(``BASE``, array ``A``-``E``, ``IRRADIANCE``, and ``TRACKING``). 
``A`` to ``E`` represent solar panel arrays at the solar farm. The irradiance module stores 
solar radiation data recorded by pyronometers and other sensors. 

The first field of each log record is a timestamp encoded as a string. For storage
in the database, these are converted to Unix timestamps, which are integers representing 
the number of seconds since 00:00 January 1, 1970, UTC. Also, in order to speed
searches, additional date and time fields are created and database indexes are defined 
for these: ``YEAR``, ``MONTH``, ``DAY``, ``HOUR``, ``MINUTE``, and ``DAYOFYEAR``. 

The schemas for the database tables are stored in separate SQL ``CREATE TABLE`` 
statements. An SQL script file that can be used to create a database of the 
appropriate schema is stored in the ``apollo.assets.sql`` directory. 

The routines for converting and inserting log data into a database assume that 
the log data is already in the correct database format, or else is in one of 
the two log formats described above and so can be converted to the database format. 
Data is read from the log files using :func:`pandas.read_csv` and so also 
must be  in a format compatible with that. GZipped csv files can be used. 

An empty database of the appropreate schema can be created using :func:`.init_db`.
The SQL statements used to create the database can be obtained using :func:`get_create_sql`

Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We create an empty database for solar farm log data using :func:`init_db`, 

>>> dbfile = "C:/test/solar_farm.db"
>>> init_db(dbfile)


The below set of statements converts a set of ``csv`` files, taken from  the 
REAPR site and stored  in ``C:/test/reapr_in``, to make the set 
compatiable with the internal database format. Each output file is saved
to the ``C:/test/reapr_out`` directory. Afterwards, the converted files 
are inserted into the ``IRRADIANCE`` table of the ``solar_farm.db`` database. 

We test that the insertion was successful by querying the database, printing out the 
year, month, and days covered. 

>>> x = REAPRWrapper()
>>> indir = "C:/test/reapr_in"
>>> outdir = "C:/test/reapr_out"
>>> x.convert_csv_dir(indir, outdir)
>>> init_db(dbfile)
>>> x.insert_dir(dbfile,outdir, table = 'IRRADIANCE', convert=False)
>>> import apollo.db.dbapi as dbapi
>>> dbh = dbapi.DBHandler(dbfile)
>>> dbh.connect()
    Out[22]: <sqlite3.Connection at 0x221cfe6b110>
>>> results =  dbh.execute('select DISTINCT YEAR, MONTH, DAY from IRRADIANCE;')
>>> for row in results: 
>>>     print(row)
    (2018, 1, 1)
    (2018, 1, 2)
    (2018, 1, 3)
    (2018, 1, 4)
    (2018, 1, 5)
    (2018, 1, 6)
    (2018, 1, 7)
    (2018, 1, 8)
    (2018, 1, 9)
    (2018, 1, 10)
    (2018, 1, 11)
    (2018, 1, 12)
    (2018, 1, 13)
    (2018, 1, 14)
    (2018, 1, 15)
    (2018, 1, 16)
    (2018, 1, 17)
    (2018, 1, 18)
    (2018, 1, 19)
    (2018, 1, 20)
    (2018, 1, 21)
    (2018, 1, 22)
    (2018, 1, 23)
    (2018, 1, 24)
    (2018, 1, 25)

The code does something similar for a set of compressed log files. Here, however, 
the conversion is specified as part of the ``insert_dir`` invocation. 

After insertion, we see that data for several more years was added to the 
database. 

>>> x2 = SolarLogWrapper()
>>> indir = "C:/test/gz_in"
>>> x2.insert_dir(dbfile,indir, table = 'IRRADIANCE', convert=True)
>>> dbh.connect()
    Out[22]: <sqlite3.Connection at 0x221d03c8d50>
>>> results =  dbh.execute('select DISTINCT YEAR, MONTH from IRRADIANCE;')
>>> for row in results: 
>>>     print(row)
    (1970, 1)
    (2016, 7)
    (2016, 8)
    (2016, 9)
    (2016, 10)
    (2016, 11)
    (2016, 12)
    (2017, 1)
    (2017, 2)
    (2017, 3)
    (2017, 4)
    (2017, 5)
    (2017, 6)
    (2017, 7)
    (2017, 8)
    (2017, 9)
    (2017, 10)
    (2017, 11)
    (2017, 12)
    (2018, 1)

""" 
import numpy as np
import pandas as pd
import pkg_resources
import os
import datetime
import gzip
import logging
import apollo.db.dbapi as dbapi
import apollo.assets.api as assets

from pathlib import Path

logger = logging.getLogger(__name__)



# Create db script, lazily read from the assets folder at runtime. 
_SQL = None
# path to create db sql. 
_SOLAR_FARM_SQL = 'assets/sql/solar_farm.sql'

# The SQL script for creating the database. Observations will be stored 
# in a table called OBS
def get_create_sql():
    """Read the GAEMN SQL create table script from the assets folder
    """
    global _SQL
    if _SQL is None:
        _SQL = pkg_resources.resource_string(
            'apollo',
            _SOLAR_FARM_SQL
        ).decode("utf-8")
    return _SQL



def init_db(db_file):
    """Initialize the solar farm database schema, using the given database file. 
    
    This will create a SQLite database file for solar farm log data 
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
        dbh = dbapi.DBHandler(db_file)
        conn = dbh.connect()
        logging.info("connected...")
        logging.info("invoking script...")
        conn.executescript(get_create_sql())
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

class REAPRWrapper:
    """Converts csv data from the REAPR site to the internal database format. 
    
    The timestamp of the REAPR file is converted into an Unix timestamp. 
    Also, ``YEAR``, ``MONTH``, ``DAY``, ``HOUR``, ``MINUTE``, ``DAYOFYEAR``, as well as 
    three additional error code columns (``CODE1``, ``CODE2``, ``CODE3``) 
    are added after the timestamp to make the data in the REAPR files compatible
    with the ealier log files. 

    Attributes:
        date_format (str) The format to use when parsing dates in input files. 
        use_headers (boolean): Inlcude headers in output (csv) files. 
    """
    
    def __init__(self,date_format='%m/%d/%Y %I:%M:%S %p', use_headers = True,  chunksize = 100000):
        """Creates an instance of the wrapper, storing the date format 
        (if any) to use when parsing timestamps in log records.  
        
        Arguments:
            date_format (str):  The format of input record timestamps. 
            use_headers (boolean): Inlcude headers in output (csv) files. 
        
        """
        self.date_format = date_format
        self.use_headers = use_headers
        self.chunksize = chunksize
        # add CODE1 CODE2 CODE3 columns when coverting to db format. 
        self.add_codes = True 


    def insert_csv(self, db_file,csv_to_insert,table=None, usetemp=True, convert=True):
        """Inserts a csv file into the given database table. 
        
        If no table is specified, an attempt is made to infer the table from 
        the csv header row. Becuase of this, the file must contain a header row. 
        
        The csv is first read into a :class:`pandas.DataFrame` object. If 
        ``usetemp=True``, then the dataframe is first inserted into a 
        temporary database table (with name ``_TEMP`` appended to the end of 
        the table name) and then copied to ``table``. The records are then deleted 
        from the temporary table. 
        
        The temporary table is needed to avoid records with duplicate keys 
        throwing errors. :func:`pandas.DataFrame.to_sql` does not appear to 
        automatically ignore duplicate keys. 
        
        If ``convert==True``, then the input csv will be transformed first, 
        converting the record timestamp to an integer and adding columns for 
        ``YEAR``, ``MONTH``, ``DAY``, ``HOUR``, ``MINUTE``, ``DAYOFYEAR`` as 
        well as 3 error code columns. 
 
        Arguments:
            db_file (str): The database file to insert into. 
            csv_to_insert (str): The csv file to insert. 
            table (str): The name of the table to insert into. 
            usetemp (boolean): Inserted into a temporary table first, then copy it to the final table. 
            convert (boolean): Convert the timestamps and add columns for year, month, etc.
        
        """
        dbh = dbapi.DBHandler(db_file)
        dbh.connect()
        for chunk in pd.read_csv(csv_to_insert, converters={0:self.parse_date}, chunksize=self.chunksize):
            try:
                if convert:
                    df = self.convert_df(chunk)
                else:
                    df = chunk.copy()
                if table == None:
                    module = self._infer_module(csv_to_insert,df)
                    table = module
                else:
                    module = table
                if usetemp:
                    temp_table = module+"_TEMP"
                    dbh.insert_dataframe(df, temp_table)
                    dbh.copy_table(temp_table,module)
                    dbh.execute(f"DELETE FROM {temp_table}", True)
                else:
                    dbh.insert_dataframe(df, module)
            except Exception as e:
                logger.error(f"apollo.server.converters.insert_csv: Error processing file {csv_to_insert}.\nPerhaps a data mismatch? {e}")
        dbh.close()

    def insert_dir(self, db_file,file_dir, table = None, convert=True):
        """Process a directory (of files in REAPR format), inserting its files into the appropriate database table. 
        
        When adding a file, it is converted from csv file into a pandas.DataFrame.
        Some columns are transformed and others are added. 
        The dataframe is first inserted into a temporary database table and then 
        copied to the final module table (the temporary table is then purged). 
        The temporary table is needed to avoid duplicates throwing errors. 
        (df.to_sql() does not appear to allow automatic 
        ignoring of duplicates). 
        
         Arguments:
             db_file (str or pathlib.Path): The database to insert into.
             file_dir (str or pathlib.Path): The directory of input files to process.
             table (str): The name of the table to insert into.
             convert (boolean): Indicates whether the input should be preprocessed before insertion. 
        """
        
        counter = 0                
        for root, dirs, files in os.walk(file_dir):
            for filename in files:
                infile = Path(root)/filename
                logging.info(f"processing: {infile}")
                try:
                    self.insert_csv(db_file, infile, table=table, convert=convert)
                    logging.info(f"Finished: {infile}")
                except Exception as e:
                    logging.ERROR(f"apollo.server.converters.insert_dir: Error processing file {infile}"+str(e))
                counter = counter + 1
                if counter % 10 == 0:
                    logging.info(f"files processed: {counter}")


    
    def convert_csv_file(self, infile, outfile, compression='gzip'):
        """Use :func:`REAPRWrapper.read_csv` to read a csv file, parsing the 
        date and making the format consistent with the internal database format. 
        Outputs the results to another csv file (possibly compressed). 
        
        The timestamp of the REAPR file is converted into an integer. Also, 
        ``YEAR``, ``MONTH``, ``DAY``, ``HOUR``, ``MINUTE``, ``DAYOFYEAR``, as 
        well as code columns (``CODE1``, ``CODE2``, ``CODE3``) 
        are added after the timestamp.
        
        Chunking is performed; a chunk of rows is read in and converted, and the
        result is written to the output file. 
        
        Arguments:
            infile (str): The name of the log file to convert.
            outfile (str): The name of the output file.
            compression (str): Whether the output should be compressed. Default value is 'gzip'
       """
        first = True
        import csv
        with open(outfile, 'w', newline='') as csvfile:
            wr = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for chunk in pd.read_csv(infile, converters={0:self.parse_date}, chunksize=self.chunksize):
                try:
                    df = self.convert_df(chunk)
                    if first:
                        cols = list(df.columns) 
                        wr.writerow(cols)
                        rows = df.values.tolist()
                        for row in rows:
                            wr.writerow(row)
                        #df.to_csv(outfile, index = None, header=self.use_headers)
                    else:
                        rows = df.values.tolist()
                        for row in rows:
                            wr.writerow(row)
                        #df.to_csv(outfile, index = None, header=False)
                except Exception as e:
                    print(e)
                first = False

    def convert_df(self, df):
        """Convert the :class:`pandas.DataFrame` to a format consistent with 
        the internal database. 
        
        During the format conversion, the following are performed: 
            
        * The first column is renamed ``TIMESTAMP`` and the values are \
        converted to Unix integer timestamps. It is assumed that the timestamp\
        is in the first column of the log file. 
        * ``YEAR``, ``MONTH``, ``DAY``, ``HOUR``, ``MINUTE``, ``DAYOFYEAR`` \
        fields are added to speed SELECT queries based on day, month, etc. 
        * Dummy columns ``CODE1``,``CODE2``,``CODE3`` are added after the timestamp to \
        conform with the log data provided by the ftp client. 
        
        Arguments:
            df (:class:`pandas.DataFrame`):
                A dataframe to convert. 
            convert (boolean):
                inddicates whether the data should be converted.
        
        Returns:
            :class:`pandas.DataFrame`: The converted dataframe. 
       """
        # load data from csv into pandas dataframe

        dates = pd.DatetimeIndex(df.iloc[:,0])
        
        # rename columns by removing "-"
        column_names = df.columns.values.tolist()
        for i in range(len(column_names)):
            df.rename(columns={column_names[i]:column_names[i].replace("-","")}, inplace=True)
                
        # add 3 code columns at positions 1,2,3 (done for compatibility with ftp data format)
        if self.add_codes:
            df.insert(1, "CODE3", 0) 
            df.insert(1, "CODE2", 0) 
            df.insert(1, "CODE1", 0) 
    
        # add more date-time fields, for faster query execution
        df.insert(1, "DAYOFYEAR",   dates.dayofyear)
        df.insert(1, "MINUTE",      dates.minute)
        df.insert(1, "HOUR",        dates.hour)
        df.insert(1, "DAY",         dates.day) 
        df.insert(1, "MONTH",       dates.month) 
        df.insert(1, "YEAR",        dates.year) 
        

        df.insert(1, "TIMESTAMP", df.iloc[:,0].astype(np.int64)/int(1e9))
        df.drop([df.columns[0]] ,  axis='columns', inplace=True)
        
        return df

    def convert_csv_dir(self, file_dir, out_dir, compression='gzip'):
        """Use :func:`REAPRWrapper.convert_csv_file` to convert all csv/gz files in the 
        specified directory into the internal database format, writing new 
        files to an output directory. 
        
        Arguments:
            file_dir (str): The name of the directory containing the files to process.
            outfile (str): The name of the output directory. 
            compression (str): Indicates whether the output should be compressed. Default value is `'gzip'`
       """
        logger.debug(f"Converting {file_dir}")        
        for root, dirs, files in os.walk(file_dir):
            for filename in files:
                if compression == 'gzip' and not filename.endswith('.gz'):
                    gz = filename + '.gz'
                    outfile = Path(out_dir)/gz
                else:
                    outfile = Path(out_dir)/filename
                try:
                    self.convert_csv_file(Path(root)/filename,outfile,compression=compression)
                except Exception as e:
                    logger.error(f"apollo.server.converters.convert_csv_dir: Error processing file {filename}. {e}")


    def parse_date(self, timestamp):
        """Constructs a datetime.datetime object from the given input. 
        Uses the stored datetime format to parse input timestamps. 
        
        Arguments:
            timestamp (str): The timestamp to convert. 
        Returns:
            :class:`datetime.datetime`: 
                A datetime representation of the timestamp. 
        """
        return datetime.datetime.strptime(timestamp, self.date_format)
    

    def _infer_module(self, in_filename, df):
        """Examine the file name or dataframe to infer the corresponding database table.
        
        When dealing with csv files from the REAPR site, column names are used. 
        For gzipped log files (without headers), the file name itself is used
        (e.g., 'mb-007' indicates the ``IRRADIANCE`` table). 
        
        Arguments:
            in_filename (str): The name of the input file. 
            df (``Pandas.DataFrame``): The dataframe to process
        
        Returns:
            str: 
                One of "BASE", "A","B","C","D","E","IRRADIANCE","TRACKING", or ``None``.
      
        """
        if 'UGAMET01WINDSPD' in df.columns:
            return "BASE"
        if 'UGAAINV01INVSTATUS' in df.columns:
            return "A"
        if 'UGABINV01INVSTATUS' in df.columns:
            return "B"
        if 'UGACINV01INVSTATUS' in df.columns:
            return "C"
        if 'UGADINV01INVSTATUS' in df.columns:
            return "D"
        if 'UGAEINV01INVSTATUS' in df.columns:
            return "E"
        if 'UGAAPOA1IRR' in df.columns:
            return "IRRADIANCE"
        if 'UGAATRACKER01AZMPOSDEG' in df.columns:
            return "TRACKING"
        return None
    
   
class SolarLogWrapper(REAPRWrapper):
    """A class for converting gzipped log files to the internal database format. 

    For roughly 1.5 years, log files from the solar farm were pushed to UGA via an FTP client. 
    The files were small and  compressed in `gz` format. 
 
    This class contains routines to unpack the log files and concatenate them
    into a single large log file. 
    
    It is assumed that all files for a given module (mb-001 to mb-008) are in 
    a separate subdirectory (bearing the name of the module. All files in the 
    subdirectory are processed and inserted into the table specified. 

    Attributes:
        date_format (str): The format (e.g., '%m/%d/%Y %I:%M:%S %p') to use when 
            parsing timestamps in input files. 
        use_headers (boolean): Indicates whether headers should be written to 
            output (csv) files. By default, headers are not written. 
    """

    def __init__(self,date_format="'%Y-%m-%d %H:%M:%S'",  chunksize = 100000):
        """Create an instance of the wrapper, storing the provided date format (if any). 
        
        The date format indicates how timestamps in log records are to be processed. 
        
        It is assumed that there are no headers in the log files. None are written
        to output. 
        
        Arguments:
            date_format (str): The format of input record timestamps. 
        
        """
        self.date_format = date_format
        self.use_headers = False
        self.chunksize = chunksize


    def convert_df(self, df):
            """Convert the :class:`pandas.DataFrame` to a format consistent with 
            the internal database. 
            
            During the format conversion, the following are performed: 
                
            * The first column is renamed ``TIMESTAMP`` and the values are \
            converted to Unix integer timestamps. It is assumed that the timestamp\
            is in the first column of the log file. 
            * ``YEAR``, ``MONTH``, ``DAY``, ``HOUR``, ``MINUTE``, ``DAYOFYEAR`` \
            fields are added to speed SELECT queries based on day, month, etc. 
            * Dummy columns ``CODE1``,``CODE2``,``CODE3`` are added after the timestamp to \
            conform with the log data provided by the ftp client. 
            
            Arguments:
                df (:class:`pandas.DataFrame`):
                    A dataframe to convert. 
                convert (boolean):
                    inddicates whether the data should be converted.
            
            Returns:
                :class:`pandas.DataFrame`: The converted dataframe. 
           """
            # load data from csv into pandas dataframe
            dates = pd.DatetimeIndex(df.iloc[:,0])
            # add more date-time fields, for faster query execution
            df.insert(1, "DAYOFYEAR", dates.dayofyear)
            df.insert(1, "MINUTE", dates.minute)
            df.insert(1, "HOUR", dates.hour)
            df.insert(1, "DAY", dates.day) 
            df.insert(1, "MONTH", dates.month) 
            df.insert(1, "YEAR", dates.year) 
            # add a unix representation of the timestamp (done because SQLite doesn't have a date datatype).
            df.insert(1, "TIMESTAMP", df.iloc[:,0].astype(np.int64)/int(1e9))
            #df.drop(columns=[0],inplace=True)
            df.drop([df.columns[0]] ,  axis='columns', inplace=True)
            return df

    def _infer_module(self, in_filename, df):
        """Examine the file name to infer the database table corresponding 
        to the dataframe.

        Arguments:
            in_filename (str or pathlib.Path): The name of the source csv file.
            df (pandas.DataFrame): The dataframe to process
        
        Returns:
            str: 
                One of "BASE", "A","B","C","D","E","IRRADIANCE","TRACKING", or ``None``.
      
        """
        
        in_filename= str(in_filename)
        
        if 'mb-001' in in_filename:
            return "BASE"
        if 'mb-002' in in_filename:
            return "A"
        if 'mb-003' in in_filename:
            return "B"
        if 'mb-004' in in_filename:
            return "C"
        if 'mb-005' in in_filename:
            return "D"
        if 'mb-006' in in_filename:
            return "E"
        if 'mb-007' in in_filename:
            return "IRRADIANCE"
        if 'mb-008' in in_filename:
            return "TRACKING"
        return None

    def concat_logs(self,in_dir, out_file):
        """Concatenates multiple csv files into one large file. The contents 
        will be compressed using ``gzip``
        
        Each file in the specified directory is read and 
        concatenated into a single file bearing the specified output file name. 
        
        All files in the input directory are used (no filtering of invalid formats is performed).
        It is assumed that the csv files do not contain header information.
        
        It would likely be faster to concatenate files using something other than
        Python. This method is included soley as a convenience. 
        
        Arguments:
            in_dir (str): The directory containing the gzipped files to concatenate. 
            out_file (str): The output file to generate.
        """
        error_count = 0; 
        with gzip.open(out_file, 'wb') as outfile:
            for root, dirs, files in os.walk(in_dir):
                logger.info(f"\nProcessing: {in_dir}")
                counter = 0
                for filename in files:
                    counter = counter + 1
                    try:
                        f = gzip.open(in_dir/filename, 'rb')
                        outfile.write(f.read())
                        f.close()
                    except Exception as err:
                        logger.error(f"Error with: {err}")
                        error_count = error_count + 1
                    if counter % 1000 == 0:
                        logger.info(f"Files processed: {counter}, error={error_count}")
            logger.info(f"Files processed: {counter}, error={error_count}")            
