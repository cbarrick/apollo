"""A command line script for creating an empty SQLite database for 
solar farm log data.

Once created, the database is intended to be populated by log data from the UGA solar farm.

A set of 8 tables is created, one for each of the 8 modules at the solar farm site. 
Internally, the  method :func:`apollo.db.converters.init_solar_db` is invoked.
The SQL script for creating the database is stored in :mod:`apollo.db.converters`.

If the database file to create already exists, nothing is done.:: 

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

"""
import apollo.db.converters as converters
import apollo.db.gaemn as gaemn
import argparse
import logging
from pathlib import Path


if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser(description="""Creates an empty a SQLite database for the solar farm log data. 
                                         """)
    parser.add_argument('db', metavar='DB', type=str,help='the database file to create.')
    parser.add_argument('--log', type=str, default='INFO', help='Sets the log level. One of INFO, DEBUG, ERROR, etc. Default is INFO')
    parser.add_argument('-g', '--gaemn',  action='store_true',help='create a GAEMN format database.')    
    
    args = parser.parse_args()
    logging.basicConfig(format='[{asctime}] {levelname}: {message}', style='{', level=args.log)
    logging.info(" * db:" +str(args.db))
    
    dbfile = Path(args.db)
    
    if args.gaemn:
        res = gaemn.init_gaemn_db(dbfile)
    else:
        res = converters.init_solar_db(dbfile)
    
    if(not res):
        logging.info("File already exists! Nothing done.")
