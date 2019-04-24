# -*- coding: utf-8 -*-
'''
A command line script for inserting csv files into the solar farm SQLite database.

This script can be used to insert either gzipped log files or else csv files 
downloaded from the REAPR website into an SQLite database of matching format. 
Either a single file or a directory of files can be inserted. 

The script can attempt to infer the appropriate table from the csv headers of REAPR files, 
but a table must be specified for the gzipped log files. 

See :mod:`apollo.db.converters` for more information.:: 

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



'''

from pathlib import Path
import argparse
import logging
import apollo.db.converters as converters
import sys
def _config_from_args():
    parser = argparse.ArgumentParser(description="Utility function for inserting logged data from the solar farm into an SQLite databas. "\
                                     +"Input is a csv file in either the REAPR or gzipped log format. "\
                                     +"By default, the data is converted (year, month, day, hour, minute, and dayofyear columns are added). "\
                                     +"Alternatively, the conversion can be skipped. ")
    
    parser.add_argument('-c', '--noconvert',  action='store_true',help='do not convert the input file or directory contents.')
    parser.add_argument('-u', '--notemp',  action='store_true',help='do not insert the data into a temporary database table before copying it to the final destination.')
    parser.add_argument('-f', '--format',   metavar='format', type=str, choices=['log', 'reapr'], dest='format', help='the format of the input file (gz "log" or "reapr" file).')
    parser.add_argument('-i', '--in',       metavar='in', type=str, dest='infile',default=None,required=True,help='the file or directory of files to insert into the database.')
    parser.add_argument('-b', '--db',       metavar='db', type=str, dest='dbfile',default=None,required=True,help='the target database to insert into.')
    parser.add_argument('-t', '--table',    metavar='table', type=str, dest='table',default=None,help='the database table to insert into.')
    parser.add_argument('--log', type=str, default='INFO', help='Sets the log level. One of INFO, DEBUG, ERROR, etc. Default is INFO')
    
    args = parser.parse_args()

    logging.basicConfig(format='[{asctime}] {levelname}: {message}', style='{', level=args.log)

    if not args.noconvert and not args.format in ['log', 'reapr']:
        parser.error ("if converting input, then --format must be 'log' or 'reapr' .")
    
    logging.info("database insert...")
    if args.format:
        logging.info(" * format:"+ str(args.format))
    logging.info(" * in:" + str(args.infile))
    logging.info(" * db:" + str(args.dbfile))
    logging.info(" * table:"+ str(args.table))
    logging.info(" * no convert:"+ str(args.noconvert))
    logging.info(" * no temp:"+ str(args.notemp))
    return args

if __name__ == "__main__":
    args = _config_from_args()
    usetemp = not args.notemp
    convert = not args.noconvert
    handler = converters.REAPRWrapper()
    infile = Path(args.infile)
    dbfile = Path(args.dbfile)
    if args.format == 'log':
        handler = converters.SolarLogWrapper()
    if not infile.exists():
        logging.info(" * input file or directory does not exist! Exiting.")
        sys.exit()
    if not dbfile.exists():
        logging.info(" * database file does not exist! Exiting.")
        sys.exit()
    
    if infile.is_dir():
        handler.insert_dir(str(dbfile), infile, table=args.table,  convert=convert)
    else:
        handler.insert_csv(str(dbfile), infile, table=args.table,usetemp=usetemp, convert=convert)