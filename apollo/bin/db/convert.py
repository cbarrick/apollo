# -*- coding: utf-8 -*-
"""A command line script to convert csv files of solar farm log data. 

This script can be used to convert either gzipped log files or else csv files 
downloaded from the REAPR website for use in an SQLite database of matching 
format. Either a single file or a directory of files can be converted.

See :mod:`apollo.db.converters` for more information.:: 

    $ python -m apollo.db.convert --format reapr  --in "C:/test/reapr_in" --out "C:/test/reapr_out"
    INFO:  * format:reapr
    INFO:  * in:C:/test/reapr_in
    INFO:  * out:C:/test/reapr_out

    $ python -m apollo.db.convert --format log  --in "C:/test/gz_in" --out "C:/test/gz_out"
    INFO:  * format:log
    INFO:  * in:C:/test/gz_in
    INFO:  * out:C:/test/gz_out

"""
import argparse
import apollo.datasets.converters as converters
import os
from pathlib import Path
import logging
import sys


def _ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def _config_from_args():
    parser = argparse.ArgumentParser(description="Utility function for converting logged data from the solar farm. "\
                                     +"Input is a csv file or directory of files in either the REAPR or gzipped log format. "\
                                     +"The input is converted by standardizing timestamps and adding year, month, day, hour, minute, and dayofyear columns. "\
                                     +"The output files are intended to be later inserted into a database table with matching format.")
    
    parser.add_argument('-f', '--format',   metavar='format', type=str, choices=['log', 'reapr'], dest='format', help='the format of the input file (gz "log" or "reapr" file).')
    parser.add_argument('-i', '--in', metavar='in', type=str, dest='infile',default=None,required=True,help='the file or directory to convert.')
    parser.add_argument('-o', '--out', metavar='out', type=str, dest='outfile',default=None,required=True,help='the file or directory of files resulting from the conversion.')
    parser.add_argument('-a', '--dates', metavar='out', type=str, dest='dates',help='format of the input dates/timestamps.')
    parser.add_argument('--log', type=str, default='INFO', help='Sets the log level. One of INFO, DEBUG, ERROR, etc. Default is INFO')
    
    args = parser.parse_args()

    logging.basicConfig(format='[{asctime}] {levelname}: {message}', style='{', level=args.log)
    
    if not args.format in ['log', 'reapr']:
        parser.error ("--format must be 'log' or 'reapr' .")
    logging.info(" * format:" +str(args.format))
    logging.info(" * in:"+str(args.infile))
    logging.info(" * out:"+str(args.outfile))
    return args


if __name__ == "__main__":
    args = _config_from_args()
    handler = None
    if args.format == 'log':
        if args.dates:
            handler = converters.SolarLogWrapper(args.dates)
        else:
            handler = converters.SolarLogWrapper()
    else:
        if args.dates:
            handler = converters.REAPRWrapper(args.dates)
        else:
            handler = converters.REAPRWrapper()
    infile = Path(args.infile)
    if not infile.exists():
        logging.info(" * input file or directory does not exist! Exiting.")
        sys.exit()

    if infile.is_dir():
        outdir = Path(args.outfile)
        _ensure_dir_exists(outdir)
        handler.convert_csv_dir(infile, outdir)
    else:
        handler.convert_csv_file(infile, args.outfile)
