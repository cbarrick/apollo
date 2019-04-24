"""A script for converting a directory of log files into the internal 
solar farm SQLite database format and then inserting the data into the appropriate
tables. 

This script is not intended to be run in the normal course of events, and it requires
customization to run. It should only be run if users need to recreate the solar farm database 
and populate it with a large cache of historical log data. 

It is more common that :mod:`apollo.db.insert` or :mod:`apollo.db.converters`  
would be run to insert additional log data into a preexisting database. 
Specifically, :mod:`apollo.db.insert` should be run to update the database periodically. 

Users will need to verify that this script is set up correctly to use their 
log files. 
"""

import apollo.db.converters as converters
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def process_gz_files(tables, gz_path, db_file):
    """Process a list of tables, inserting a corresponding  (gzipped) log file
    into the appropriate database table. 

    Arguments:
        tables (list): A list of table names. 
        gz_path (str or pathlib.Path): The directory storting the gzipped log files. 
        db_file (str or pathlib.Path): The database to insert into.
    """    
    handler = converters.SolarLogWrapper()
    for table in tables: 
        table_file = table + ".csv.gz"
        logger.info(table_file)
        infile = Path(gz_path)/table_file
        handler.insert_csv(db_file, infile, table=table,usetemp=True, convert=True)

def process_reapr_files(tables, reapr_path, db_file):
    """Process a list of tables, inserting corresponding  REAPR log files
    into the appropriate database table. 

    Arguments:
        tables (list): A list of table names. 
        reapr_path (str or pathlib.Path): The directory storting the REAPR log files. 
        db_file (str or pathlib.Path): The database to insert into.
    """    
    for table in tables: 
        logger.info(Path(reapr_path)/table)
        process_reapr(dbfile, Path(reapr_path)/table, table=table,  convert=True)


def process_reapr(db_file,file_dir, table = None, convert=True):
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
                rbh = converters.REAPRWrapper()
                rbh.insert_csv(db_file, infile, table=table, convert=convert)
                logging.info(f"Finished: {infile}")
            except Exception as e:
                logging.ERROR(f"Error processing REAPR file {infile}"+str(e))
            counter = counter + 1
            if counter % 10 == 0:
                logging.info(f"files processed: {counter}")

if __name__ == "__main__":
    
    gz_file_dir = Path("I:/Solar Radition Project Data April 2018/bigdata/in/gz")
    reapr_file_dir = Path("I:/Solar Radition Project Data April 2018/bigdata/in/REAPR")
    dbfile = "I:/Solar Radition Project Data April 2018/bigdata/out/solar_farm_sqlite.db"
    tables = ["BASE", "A", "B","C","D","E","IRRADIANCE","TRACKING"]
    logging.basicConfig(format='[{asctime}] {levelname}: {message}', style='{', level="INFO")

    # flags for running parts of the script.     
    # change to run one or more parts. 
    init_db = True
    insert_gz = True
    insert_reapr = True

    # create the database file
    if init_db:
        logging.info("Creating database {dbfile}")
        converters.init_db(dbfile)
    
    # process the gz log files. 
    if insert_gz:
        logging.info("Processing gz files")
        process_gz_files(tables, gz_file_dir, dbfile)

    # process the reapr log files. 
    if insert_reapr:
        logging.info("Processing reapr files")
        process_reapr_files(tables, reapr_file_dir, dbfile)
