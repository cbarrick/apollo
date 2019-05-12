# -*- coding: utf-8 -*-
"""Defines classes for handling HTTP requests to the solar farm or other databases. 

The subclasses of :class:`ServerRequestHandler` implement 
:func:`ServerRequestHandler.handle_request`, which produces formatted output to 
:mod:`flask` server requests. The requests encode a set of attribute-value pairs, 
and these are used to construct an SQL ``SELECT`` query to a database. 
A sample collection of pairs is shown below. 

.. code-block:: python   

    ImmutableMultiDict([
            ('source', 'solar_farm'), # database to use. 
            ('site', 'IRRADIANCE'),       # database table
            ('schema', 'solar_farm'),     # schema used to format results. 
            ('start', '1483160400000'),   # start timestamp in ms since 1/1970
            ('stop', '1483246800000'),    # end timestamp in ms since 1/1970
            ('attribute', 'UGAAPOA1IRR'), # table column to retrieve
            ('attribute', 'UGAAPOA2IRR'), # table column to retrieve
            ('attribute', 'UGAAPOA3IRR'), # table column to retrieve
            ('groupby', 'yearmonthdayhourmin'), # SQL GROUP BY information
            ('statistic', 'AVG'),         # group statistic to retrieve
            ('statistic', 'MIN'),         # group statistic to retrieve
            ('statistic', 'MAX')])        # group statistic to retrieve

This particular request asks for the minimum, maximum, and average values of 
three irradiance  variables in the ``IRRADIANCE`` table of the solar farm database. The 
statistics are calcuated over minute-long groups of values. Start and stop
times are encoded as the number of milliseconds since 00:00 January 1, 1970 UTC. 

The request generates the following SQL query, which is then posed to the database
``solar_farm.db``. The database resides in the Apollo Data Explorer database 
directory. The name specified by ``source`` is the database queried.

.. code-block:: sql

    SELECT  
        MIN(TIMESTAMP), AVG(UGAAPOA1IRR),AVG(UGAAPOA2IRR),AVG(UGAAPOA3IRR),
        MIN(UGAAPOA1IRR),MIN(UGAAPOA2IRR),MIN(UGAAPOA3IRR),
        MAX(UGAAPOA1IRR),MAX(UGAAPOA2IRR),MAX(UGAAPOA3IRR) 
    FROM 
        IRRADIANCE 
    WHERE 
        TIMESTAMP >= 1483160400 AND TIMESTAMP <= 1483246800 
    GROUP BY 
        YEAR, MONTH, DAY, HOUR, strftime('%M', datetime(TIMESTAMP, 'unixepoch'))  
    LIMIT 50000;

The database table contains the columns ``TIMESTAMP``, ``YEAR``, 
``MONTH``, ``DAY``, and ``HOUR`` in addition to ``UGAAPOA1IRR``, ``UGAAPOA2IRR``, and ``UGAAPOA3IRR``. 
The Apollo Data Explorer database access routines are designed to work with this 
specific database format.
In particular, the routines assume that the columns related to dates and times are defined. 
Substituting another database would require altering both the server-side and 
client-side code. 

As is done in this example, it is possible to group results. The grouping options are: 
``yearmonthdayhourmin``, ``yearmonthdayhour``, ``yearmonthday``, ``yearmonth``, 
``monthofyear``, ``dayofyear``, ``dayhourofyear``, ``proc_yearmonthdayhourmin``,
``proc_yearmonthdayhour``, ``proc_yearmonthday``, and ``proc_yearmonth``. For 
options preceded with ``proc_``, :mod:`pandas` is used to group values and 
compute statistics, but for the other options, the SQLite database engine is used. Note
that :mod:`pandas` permits a wider variety of statistics to be computed than SQLite
(SQLite  permits little more than max, min, avg, and count). 

The formatted output is produced with the aid of a schema file (in this case named 
``solar_farm.json``) which holds human-readable descriptions for each of the 
result columns. The formatted output will something like the following 
(most of the row data has been omitted).

.. code-block:: python

    	{
    	"columns":[
    		{"description":"Unix integer timestamp","label":"MIN TIMESTAMP","type":"datetime","units":"s"},
    		{"description":"Pyranometer PYR01: Irradiance - Instantaneous - Plane of Array Value - from Logger","label":"AVG UGAAPOA1IRR","type":"number","units":"w/m2"},
    		{"description":"Pyranometer PYR02: LICOR - Irradiance - Instantaneous - Plane of Array Value - from Logger","label":"AVG UGAAPOA2IRR","type":"number","units":"w/m2"},
    		{"description":"Pyranometer PYR03: LICOR  Irradiance - Instantaneous - Plane of Array Value - from Logger","label":"AVG UGAAPOA3IRR","type":"number","units":"w/m2"},
    		{"description":"Pyranometer PYR01: Irradiance - Instantaneous - Plane of Array Value - from Logger","label":"MIN UGAAPOA1IRR","type":"number","units":"w/m2"},
    		{"description":"Pyranometer PYR02: LICOR - Irradiance - Instantaneous - Plane of Array Value - from Logger","label":"MIN UGAAPOA2IRR","type":"number","units":"w/m2"},
    		{"description":"Pyranometer PYR03: LICOR  Irradiance - Instantaneous - Plane of Array Value - from Logger","label":"MIN UGAAPOA3IRR","type":"number","units":"w/m2"},
    		{"description":"Pyranometer PYR01: Irradiance - Instantaneous - Plane of Array Value - from Logger","label":"MAX UGAAPOA1IRR","type":"number","units":"w/m2"},
    		{"description":"Pyranometer PYR02: LICOR - Irradiance - Instantaneous - Plane of Array Value - from Logger","label":"MAX UGAAPOA2IRR","type":"number","units":"w/m2"},
    		{"description":"Pyranometer PYR03: LICOR  Irradiance - Instantaneous - Plane of Array Value - from Logger","label":"MAX UGAAPOA3IRR","type":"number","units":"w/m2"}
    		],
    	"rows":[
                [1483142400000,0,0,0,0,0,0,0,0,0],
                [1483142460000,0,0,0,0,0,0,0,0,0],
                [1483142520000,0,0.016916667,0,0,0,0,0,0.203,0],
                [1483142580000,0,0,0,0,0,0,0,0,0],
                ...],
    	"site":"IRRADIANCE",
    	"start":1483160400000,
    	"stop":1483246800000,
    	"subtitle":"MIN,AVG,MAX",
    	"title":"TIMESTAMP,UGAAPOA1IRR,UGAAPOA2IRR,UGAAPOA3IRR",
    	"units":""
    	}

These results are returned to Apollo's Data Explorer web client and then typically 
used to produce tables or charts. 
"""

import pandas as pd
import numpy as np
import os
import datetime
import logging
from pathlib import Path
from flask import jsonify
from apollo.datasets import ga_power
import apollo.server.schemas as schemas
import apollo.storage as storage
import pytz

logger = logging.getLogger(__name__)

TIMESTAMP = "TIMESTAMP"

QUERY_SOURCE_KEY =      "source"
QUERY_SCHEMA_KEY =      "schema"
QUERY_SITE_KEY =        "site"
QUERY_ATTRIBUTE_KEY =   "attribute"
QUERY_STATISTIC_KEY =   "statistic"
QUERY_GROUPBY_KEY =     "groupby"
QUERY_START_KEY =       "start"
QUERY_STOP_KEY =        "stop"

OUTPUT_SITE_KEY =       "site"
OUTPUT_TITLE_KEY =      "title"
OUTPUT_SUBTITLE_KEY =   "subtitle"
OUTPUT_START_TIME_KEY = "start"
OUTPUT_STOP_TIME_KEY =  "stop"
OUTPUT_UNITS_KEY =      "units"
OUTPUT_ROWS_KEY =       "rows"
OUTPUT_COLUMNS_KEY =    "columns"

STATISTICS = set(["MIN", "MAX", "AVG", "MEAN", "COUNT", "SUM", "PER5","PER10","PER20", "PER25","PER50","PER75","PER90","PER95","PER99", "VAR", "STD", "VARP", "STDP"])

USE_PROC = ["proc_yearmonthdayhourmin","proc_yearmonthdayhour","proc_yearmonthday","proc_yearmonth"]


# groupby: encodes a SQL GROUP BY clause;
# fields: additional fields to put in the SELECT clause
# timestamp: encodes the time field(s) to be used in a SELECT clause. 

GROUP_BY_DICTIONARY = {
       "yearmonthdayhourmin": {
               "groupby":"GROUP BY YEAR, MONTH, DAY, HOUR, strftime('%M', datetime(TIMESTAMP, 'unixepoch'))",
               "fields":[],
               "timestamp": "MIN(TIMESTAMP)"
               },
		"yearmonthdayhour": {
                "groupby": "GROUP BY YEAR, MONTH, DAY, HOUR",
               "fields":[],
                "timestamp": "MIN(TIMESTAMP)"
                       },
		"yearmonthday": {
                "groupby":"GROUP BY YEAR, MONTH, DAY",
               "fields":[],
                "timestamp": "MIN(TIMESTAMP)"},
		"yearmonth": {
                "groupby":"GROUP BY YEAR, MONTH ORDER BY YEAR, MONTH",
               "fields":[],
                "timestamp": "MIN(TIMESTAMP)"},
		"dayhourofyear": {
                "groupby": "GROUP BY DAYOFYEAR, HOUR",
               "fields":[],
                "timestamp": "MIN(TIMESTAMP)"},
		"monthofyear": {
                "groupby": "GROUP BY MONTH ORDER BY MONTH",
               "fields":[],
                "timestamp": "MONTH"},
       "dayofyear": {
               "groupby":"GROUP BY DAYOFYEAR",
               "fields":[],
               "timestamp": "DAYOFYEAR"},
       "proc_yearmonthdayhourmin": {
               "groupby":"",
               "fields":['YEAR','MONTH', 'DAY', 'HOUR','MINUTE'],
               "timestamp": "TIMESTAMP, YEAR, MONTH, DAY, HOUR, MINUTE"},
       "proc_yearmonthdayhour": {
               "groupby":"",
               "fields":['YEAR','MONTH', 'DAY', 'HOUR'],
               "timestamp": "TIMESTAMP, YEAR, MONTH, DAY, HOUR"},
       "proc_yearmonthday": {
               "groupby":"",
               "fields":['YEAR','MONTH', 'DAY'],
               "timestamp": "TIMESTAMP, YEAR, MONTH, DAY"},
       "proc_yearmonth": {
               "groupby":"",
               "fields":['YEAR','MONTH'],
               "timestamp": "TIMESTAMP, YEAR, MONTH" 
               }
        }


def columns(source,table):
    """Retrieve the column names of a database table. 
    
    Arguments:
        source (str): The database to access. 
        table (str): The table to access. 
    
    Returns:
        list:
            A list containing the names (strings) of the columns in the table. 
    """
    dbh = None
    try:
        dbh = ga_power.DBHandler(source)
        dbh.connect()
        columns = dbh.column_names(table)
        dbh.close()
        return columns
    except Exception as e:
        if dbh:
            dbh.close()
        raise Exception(f'Error accessing columns for {table} in {source}. {e}')

def tables(source):
    """Retrieve the names of tables in a database. 
    
    Arguments:
        source (str): The database to access. 
    
    Returns:
        list:
            A list containing the names (strings) of the tables in the database. 
    """
    dbh = None
    try:
        dbh = ga_power.DBHandler(source)
        dbh.connect()
        tables = dbh.tables()
        dbh.close()
        return tables
    except Exception as e:
        if dbh:
            dbh.close()
        raise Exception(f'Error accessing tables for source {source}. {e}')




def factory(request, db_dir=None, db_file=None, row_limit = 50000):
    """Return an appropriate instance of :class:`ServerRequestHandler`, based 
    on the request parameters. 
    
    Either :class:`.SolarDBRequestHandler` or 
    :class:`.SolarDBRequestHandlerPandas` is returned. 
    
    Arguments:
        request (:class:`flask.Request`): The request encoding the database query. 
        db_dir (str or :class:`pathlib.Path`): The directory of databases to use. 
        db_file (str): The default database to use (excluding the extension ``'.db'``).
        row_limit (int): The maximum number of results. If 0, then no limit. 

    Returns: 
        :class:`ServerRequestHandler`:
            An instance of a :class:`ServerRequestHandler` subclass. 
    """
    groupby =  request.args.get(QUERY_GROUPBY_KEY, "")
    if groupby in USE_PROC:
        return SolarDBRequestHandlerPandas(db_dir=db_dir, db_file=db_file, row_limit = row_limit)
    else:
        return SolarDBRequestHandler(db_dir=db_dir, db_file=db_file, row_limit = row_limit)

class ServerRequestHandler:
    """The superclass for handling database requests. 
    
    Each subclass should override :func:`ServerRequestHandler.handle_request`.
    
    Arguments:
        db_dir (str or :class:`pathlib.Path`): The directory of databases. 
        db_file (str): The default database to use (excluding the extension).
        row_limit (int): The maximum number of results. If 0, then no limit. 
    """
    
    def __init__(self, db_dir = None, db_file = None, row_limit=50000):
        """Create an instance of the handler.
        """
        self.db_dir = Path(db_dir)
        self.db_file = db_file
        self.row_limit = row_limit
    
    def handle_request(self,request, **args):
        """Generate a response to the request. 
        
        In this case, the request, as a string, is returned. Subclasses should 
        override this method. 
        
        Arguments:
            request (:class:`flask.Request`): The request encoding the database query. 
            **args (dict): a catchall for additional information. 
        
        Returns: 
            :class:`flask.Response`:
                The response to the client. 
        """
        try:
            return str(request)
        except Exception as e:
            return 'Bad Request: '+str(e), 400
        
class SolarDBRequestHandler(ServerRequestHandler):
    """Handles requests to a database, possibly grouping results 
    using ``GROUP BY`` clauses. 
    
    This class is used to query a database, possibly grouping results
    using routines built into the datatabase software (SQLite). This is in 
    contrast to queries requiring post-processing with :mod:`pandas` or :mod:`numpy`. 
    """
    def handle_request(self, request, **args):  
        """Parses the request object and queries the specified database.
        
        Results are formatted as JSON and returned to the client.
        
        Arguments:
            request (:class:`flask.Request`): The request encoding the database query. 
            **args (dict): a catchall for additional information. 
        
            Returns: 
                :class:`flask.Response`:
                    The response to the client (generally, JSON). 
        """
        response_dictionary = self._process_args(request)
        js = jsonify(response_dictionary)
        logger.debug("QUERY COMPLETE")
        return js

    def _get_db_file(self, source, schema=None, ext=None):
        """"Attempt to locate the database file associated with the given source and schema

        The `$APOLLO_DATA/GA-POWER` directory is searched first.
        If the database is not found in that directory, then the
        `$APOLLO_DATA/databases` directory is searched.
        If the database cannot be found in that directory, then the path to
        the default database is returned.

        """
        db_name = f'{source}{ext}'
        print(db_name)
        if schema:
            path = storage.get('GA-POWER')/ schema / db_name
            if os.path.isfile(str(path)):
                return str(path)

            path = storage.get('databases') / schema / db_name
            if os.path.isfile(str(path)):
                return str(path)

        return str(self.db_file)
    
    def _process_args(self, query_request):
        """Convert a request to SQL and query a databse, returning 
        the results as a Python dictionary.
        """
        args = query_request.args
        source =        args.get(QUERY_SOURCE_KEY,None)
        schema =        args.get(QUERY_SCHEMA_KEY,None)
        table =         args.get(QUERY_SITE_KEY,None)
        attributes =    args.getlist(QUERY_ATTRIBUTE_KEY)
        statistics =    args.getlist(QUERY_STATISTIC_KEY)
        groupby =       args.get(QUERY_GROUPBY_KEY, "")
        start_raw =     args.get(QUERY_START_KEY,None)
        stop_raw =      args.get(QUERY_STOP_KEY,None)
    
        start = start_raw
        stop =  stop_raw
        
        # attempt to use 'sources.json' to get path of db 
        sources = schemas.get_sources(Path(self.db_dir), 'sources.json')
        ext = '.db' # default extension
        if sources:
            for s in sources:
                if source == s['id']:
                    if 'ext' in s:
                        ext = s['ext']
                    break
        # search for dbfile, checking schema dir then default db dir. 
        dbfile = self._get_db_file(source=source, schema=schema, ext=ext)
        if not dbfile:
            raise Exception("Data source not found for the following: " + str(source))
        
        
        attributes = self._attributes_check(dbfile, table, attributes)
        
        # filter out invalid options
        statistics = [st for st in statistics if self._statistic_check(st)]

        try:
            start = datetime.datetime.fromtimestamp(int(start)/ 1e3,tz=pytz.utc)
            stop = datetime.datetime.fromtimestamp(int(stop) / 1e3,tz=pytz.utc)
        except Exception as e:
            raise Exception(f'Incorrect start or stop time format. Found values are {start} and {stop}.\n' + str(e))
        
        logger.debug(f"QUERY: source={source}, schema={schema}, start={start}, stop={stop}, table={table},attributes={attributes}, groupby={groupby}, statistics={statistics}")
        
        columns =  self._get_sql_columns(statistics, attributes)
        where_clause =  f"WHERE TIMESTAMP >= {int(start.timestamp())} AND TIMESTAMP <= {int(stop.timestamp())}"
        timestamp, groupby_modified = self._get_select_time(groupby)
        
        row_limit = ""
        if self.row_limit > 0:
            row_limit = " LIMIT " + str(self.row_limit)

        sql = f"SELECT {timestamp}, {columns} FROM {table} {where_clause} {groupby_modified} {row_limit}"
    
        if groupby in USE_PROC:
            timestamp = "TIMESTAMP"
        
        df = ga_power.query_db(dbfile,sql)

        response_dict = self._query_format_response(schema,table, int(start_raw), int(stop_raw), timestamp,groupby,statistics,df)
        return response_dict

    def _get_select_time(self, groupby_key):
        """Using the given key, return a sql GROUP BY statement, together with a
        a name for the column to use as the timestamp. 
        """
        if groupby_key in GROUP_BY_DICTIONARY:
            groupby = GROUP_BY_DICTIONARY[groupby_key]["groupby"]
            timestamp = GROUP_BY_DICTIONARY[groupby_key]["timestamp"]
        else:
            groupby = ""
            timestamp = TIMESTAMP
        return timestamp, groupby
        
    def _get_sql_columns(self, statistics=[], attributes=[]):
        '''Constructs arguments for an SQL SELECT statement, possibly using statistics.
        
        Statistics should be a list from ["MIN", "MAX", "AVG", "COUNT"]. 
        Attributes should be a list of attributes in the specified table. 
        
        '''
        # if the stats list is not empty, construct terms for each statistic and attribute. 
        if statistics:
            return ",".join([",".join([f"{stat}("+str(att)+")" for att in attributes]) for stat in statistics])
        # otherwise join the attributes
        else:
            return ",".join(attributes)
    
    def _statistic_check(self, exp):
        '''Determines whether string ``exp`` is in the list of approved statistics (``MIN``, ``MAX``, etc.).
        
        Returns:
            ``exp`` if it is a valid expression. Otherwise it raises an exception. 
        '''
        return str(exp).upper() in STATISTICS
            
    def _attributes_check(self, dbfile, table, attributes):
        '''Determines whether string ``a`` is in the list of columns of ``table``.
        
        Returns:
            ``a`` if it is a valid expression. Otherwise it raises an exception. 
        '''
        attributes_temp = []
        try:
            handler = ga_power.DBHandler(dbfile)
            handler.connect()
            columns = handler.column_names(table)
            attributes_temp = [at for at in attributes if at in columns]
            handler.close()
        except: 
            raise Exception(f"Problem connecting to db={dbfile}, site/table={table}, attributes={attributes}")
        
        if len(attributes_temp) == 0:
            raise Exception(f"Invalid attribute list: db={dbfile}, site/table={table}, attributes={attributes}")
        return attributes_temp
    
    
    
     
    def _strip_statistics(self, name):
        """Split MIN(MONTH), for instance into to [MONTH, MIN]"""
        for stat in STATISTICS:
            if stat in name:
                name = name.replace(stat+"(","").replace(")","")
                return [name, stat]
        return [name, None]
    
    
    
    def _query_format_response(self, schema, site, start, stop, timestamp_column, groupby, statistics, df):
        """"Format the results of the database query, returning a Python 
        dictionary to be converted to ``JSON``
        """
        logger.debug("formatting response...")
        columns = []
        rows = []
        title_str = ""
        subtitle_str = ""
        unique_attribute_list = []
        unique_statistics_list = []
        if df is not None:
            # if grouped by month or day of year, time column will not be a 
            # unix timestamp, and so don't convert that column to ms. 
            if groupby == "monthofyear" or groupby == "dayofyear":
                pass
            else:
                df[timestamp_column] = df[timestamp_column] * 1e3
            
            # Done because NaN is not part of JSON. None should become null when jsonified. 
            df.replace({pd.np.nan: None}, inplace=True)
            
            # get the the actual data as a list. 
            rows = df.values.tolist()                
            
            # Make o list of attributes and their statistic modifiers.  MAX(ATTR) becomes [ATTR, MAX]
            attribute_list = [ self._strip_statistics(colname) for colname in df]
            
            # find human readable descriptions of the columns. 
            for [col_name,statistic] in attribute_list:
                
                if col_name not in unique_attribute_list:
                    unique_attribute_list.append(col_name)
                if statistic and statistic not in unique_statistics_list:
                    unique_statistics_list.append(statistic)
                    
                metadata = schemas.get_schema_data(self.db_dir, schema, site, col_name)
                statprefix = ""
                if statistic:
                    statprefix = str(statistic) + " "
                coldata = {
                        schemas.LABEL_KEY:statprefix + str(metadata[schemas.LABEL_KEY]),
                        schemas.UNITS_KEY:metadata[schemas.UNITS_KEY],
                        schemas.DESCRIPTION_KEY:metadata[schemas.DESCRIPTION_KEY],
                        schemas.CHART_DATATYPE_KEY:metadata[schemas.CHART_DATATYPE_KEY]
                        }
                columns.append(coldata)
            title_str = ",".join(unique_attribute_list)
            subtitle_str = ",".join(unique_statistics_list)
        
        # put it all together.
        return {
            OUTPUT_SITE_KEY:  site,
            OUTPUT_START_TIME_KEY: start,
            OUTPUT_STOP_TIME_KEY:  stop,
            OUTPUT_COLUMNS_KEY:  columns,
            OUTPUT_ROWS_KEY:  rows,
            OUTPUT_TITLE_KEY:  title_str,
            OUTPUT_UNITS_KEY:  "",
            OUTPUT_SUBTITLE_KEY:  subtitle_str}
        
        

############################################################################
        
class SolarDBRequestHandlerPandas(SolarDBRequestHandler):
    """Handles requests to a database, possibly post-processing results 
    using :mod:`numpy` and :mod:`pandas`. 
    
    This class is intended for database queries that require computing 
    statistics not available in the underlying database software. Currently, the following functions 
    can be specified: 
    ``MAX``, ``MIN``, ``MEAN``, ``COUNT``, ``SUM``, ``PER5``, ``PER10``, 
    ``PER20``, ``PER25``, ``PER50``, ``PER75``, ``PER90``, ``PER95``, 
    ``PER99``, ``STD``, ``STDP``, ``VAR``, ``VARP``.
    
    """
    def _get_sql_columns(self, statistics=[], attributes=[]):
        '''Constructs arguments for an SQL SELECT statement
        
        Attributes should be a list of attributes in the specified table. 
        
        In this subclass, the list of statistics is ignored (statistics are 
        dealt with elsewhere).
        '''
        return ",".join(attributes)
    
    def _strip_statistics(self, name):
        """Takes a name such as ``ATTR_MAX`` and strips the statistics suffix
        (``_MAX`` in this case) from it, returning both that attribute name and 
        its suffix. 
        """
        for stat in STATISTICS:
            stat = stat.upper()
            statsuffix = "_"+stat
            name = name.upper()
            if name.endswith(statsuffix):
                name = name.replace(statsuffix,"")
                return [name, stat]
        return [name, None]

    def _query_format_response(self, source, site, start, stop, timestamp_column, groupby, statistics, df):
        """Format the results of a query, performing grouping, returning a Python dictionary. 
        """
        logger.debug("formatting response...")
        columns = []
        rows = []
        title_str = ""
        subtitle_str = ""
        unique_attribute_list = []
        unique_statistics_list = []
        

        if df is not None:
            # if grouped by month or day of year, time column will not be a 
            # unix timestamp, and so don't convert that column to ms. 
            if groupby == "monthofyear" or groupby == "dayofyear":
                pass
            else:
                df[timestamp_column] = df[timestamp_column] * 1e3

            # Get a list of statistics to compute on each group
            statistics_func = [self._get_stat_function(x) for x in statistics]

            # perform grouping of results using pandas.
            if groupby in GROUP_BY_DICTIONARY:
                logger.debug("aggregating...")
                # form groups
                groups = df.groupby(GROUP_BY_DICTIONARY[groupby]["fields"])
                
                # use min timestamp of group as group's timestamp
                timestamps = groups['TIMESTAMP'].min();

                # compute the statistics on each group. 
                g2 = groups.agg(statistics_func)
                g2.columns = ["_".join(x) for x in g2.columns.ravel()]
                
                # drop the statistics on timestamps. 
                for col_name in g2.columns:
                    if col_name.startswith("TIMESTAMP"):
                        g2.drop([col_name], axis=1,inplace =True)
                # add the timestamp for each group
                g2.insert(0, 'TIMESTAMP',timestamps)

                # Done because NaN is not part of JSON. None should become null when jsonified. 
                g2.replace({pd.np.nan: None}, inplace=True)

                # get the attribute and stat name from the grouped data                
                attribute_list = [ self._strip_statistics(colname) for colname in g2.columns]
                
                # get the actual values. 
                rows = g2.values.tolist()                
            else:
                rows = df.values.tolist()         
                
                # get the attribute and stat name from the columns                
                attribute_list = [ self._strip_statistics(colname) for colname in df]
            
            # find human readable descriptions of the columns. 
            for [col_name,statistic] in attribute_list:
                if col_name not in unique_attribute_list:
                    unique_attribute_list.append(col_name)
                if statistic and statistic not in unique_statistics_list:
                    unique_statistics_list.append(statistic)
                    
                metadata = schemas.get_schema_data(self.db_dir, source, site, col_name)
                statprefix = ""
                if statistic:
                    statprefix = str(statistic) + " "
                coldata = {
                        schemas.LABEL_KEY:statprefix + str(metadata[schemas.LABEL_KEY]),
                        schemas.UNITS_KEY:metadata[schemas.UNITS_KEY],
                        schemas.DESCRIPTION_KEY:metadata[schemas.DESCRIPTION_KEY],
                        schemas.CHART_DATATYPE_KEY:metadata[schemas.CHART_DATATYPE_KEY]
                        }
                columns.append(coldata)

            title_str = ",".join(unique_attribute_list)
            subtitle_str = ",".join(unique_statistics_list)
        
            logger.debug("done formatting.")
        return {
            OUTPUT_SITE_KEY:  site,
            OUTPUT_START_TIME_KEY: start,
            OUTPUT_STOP_TIME_KEY:  stop,
            OUTPUT_COLUMNS_KEY:  columns,
            OUTPUT_ROWS_KEY:  rows,
            OUTPUT_TITLE_KEY:  title_str,
            OUTPUT_UNITS_KEY:  "",
            OUTPUT_SUBTITLE_KEY:  subtitle_str}
        

    def _get_stat_function(self, label):
        """Based on ``label``, retrieve a statistics function to apply to data.
        
        This is used when aggregating data using :mod:`numpy` and :mod:`pandas`.
        """
        
        def percentile(n):
            def _percentile(x):
                return np.nanpercentile(x, n)
            _percentile.__name__ = 'PER{}'.format(n)
            return _percentile

        def std():
            def _std(x):
                return np.nanstd(x, ddof=1)
            _std.__name__ = 'STD'
            return _std
        def stdp():
            def _stdp(x):
                return np.nanstd(x, ddof=0)
            _stdp.__name__ = 'STDp'
            return _stdp
  
        def var():
            def _var(x):
                return np.nanvar(x, ddof=1)
            _var.__name__ = 'VAR'
            return _var
        def varp():
            def _varp(x):
                return np.nanvar(x, ddof=0)
            _varp.__name__ = 'VARp'
            return _varp
        
        if "MAX" == label:
            return 'max'
        if "MIN" == label:
            return 'min'
        if "MEAN" == label:
            return 'mean'
        if "COUNT" == label:
            return 'count'
        if "SUM" == label:
            return 'sum'
        if "PER5" == label:
            return percentile(5)
        if "PER10" == label:
            return percentile(10)
        if "PER20" == label:
            return percentile(20)
        if "PER25" == label:
            return percentile(25)
        if "PER50" == label:
            return percentile(50)
        if "PER75" == label:
            return percentile(75)
        if "PER90" == label:
            return percentile(90)
        if "PER95" == label:
            return percentile(95)
        if "PER99" == label:
            return percentile(99)
        if "STD" == label:
            return std()
        if "STDP" == label:
            return stdp()
        if "VAR" == label:
            return var()
        if "VARP" == label:
            return varp()
        return 'min'