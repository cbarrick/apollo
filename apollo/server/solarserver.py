"""A command line script for running a :mod:`flask` server to respond to HTTP 
requests.

This is the primary server for handling queries through the Apollo Data Explorer 
web interface.  Both queries for static files and queries to the solar farm 
database (and other defined databases) are handled. 

When this script is invoked, directories for databases, HTML files, 
and supporting files should be set up correctly, and the script should be 
invoked with corresponding command line arguments.  

A sample directory configuration is shown below. ::

    home
    ├──db
    │   ├── solar_farm
    │   │   └── solar_farm.db
    │   ├── gaemn15min
    │   │   ├── ALAPAHA.db
    │   │   └── GRIFFIN.db
    │   ├── sources.json
    │   ├── solar_farm.json
    │   └── gaemn15min.json
    │   
    └──html
        └── apollo
            ├── date_utils.js
            ├── explorer.css
            ├── explorer.html
            ├── exlorer_ui.js
            ├── index.html
            ├── pvlib.html
            └── forecasts
                ├── dtree_2019-03-21 20_39_43.716000.html
                └── index.html
    
The ``html`` directory holds static HTML and other files to be served to web 
clients, while ``db`` holds databases that can be accessed by the Data Explorer. 
In this case, ``solar_farm.db``, ``ALAPAHA.db``, and ``GRIFFIN.db`` are databases 
storting historical weather observations. The file ``sources.json`` contains a list of 
all of the databases, and the Data Explorer uses this list to determine which 
databases exist and how to access them. 

The format for ``sources.json`` is shown below. Each entry records the name 
(``id``) of the database file, its file extension, the name of its corresponding 
*schema* file, and the starting and ending dates for a sample query (the dates 
are used in the web interface).

 .. code-block:: python

    [
    {"id":"solar_farm", "ext":".db", "label":"UGA Solar Farm", "schema":"solar_farm", "initial_start": "2017-01-01", "initial_stop": "2017-01-02"},
    {"id":"ALAPAHA", "ext":".db", "label":"Alapaha", "schema":"gaemn15min", "initial_start": "2013-01-01", "initial_stop": "2013-01-02"},
    ]

Files ``solar_farm.json`` and ``gaemn15min.json`` are schema files. These store 
information on the columns in database tables (their names, brief descriptions, their measurement 
units, etc.) are this information is used to format the results of queries. 
Each database should be associated with excactly one schema file. The database should be
stored in a subdirectory with the same name as the schema. 
See :class:`apollo.server.schemas` for more information. 

When this script is invoked, ``--dbdir``, ``--htmldir``, ``--dburl``, and 
``--htmlurl`` should all be specified. These specify both the directories and the URL
context to associated with them.  For instance, if if ``--htmldir c:\\web`` and 
``--htmlurl /files`` are specified, then  the following URL 
would (on a Windows machine) retrieve  ``c:\\web\\apollo\\explorer.html``.  

* ``http://127.0.0.1:5000/files/apollo/explorer.html``

If the database url is ``/query``, then 
any request to ``http://127.0.0.1:5000/query`` would be handled by the Apollo 
Data Explorer database handling routines. 

Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::
    
    usage: solarserver.py [-h] [--host IP] [--port N] [--html HTML_DIR]
        [--dbdir DB_DIR] [--dbfile DB_FILE] [--dburl dburl] [--htmlurl htmlurl] [--log LOG]
    
    arguments:
        -h, --help            show this help message and exit
        --host IP             The IP to listen on. Default is 127.0.0.1.
        --port N              The port to listen on. Default is 5000.
        --html HTML_DIR       The directory for html and static files.
        --dbdir DB_DIR        The directory storing the sqlite database(s) to use.
        --dbfile DB_FILE      The default database file to use.
        --dburl dburl         The URL to bind to database queries.
        --htmlurl htmlurl     The URL to bind to static (html) queries.
        --log LOG             Sets the log level. One of INFO, DEBUG, ERROR, etc.
                              Default is INFO.

    
Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This will start the server, listening on IP 127.0.0.1, port 5000.::
    
    $ python -m apollo.server.solarserver --host 127.0.0.1 --port 5000 --html "I:\html" --dbdir "I:\db"
    INFO:  * port:5000
    INFO:  * html:I:\html
    INFO:  * html url:/html
    INFO:  * dbdir:I:\db
    INFO:  * dbfile:default.db
    INFO:  * db url:/apollo
    INFO:  * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

"""
from flask import Flask,request,jsonify, send_from_directory
from pathlib import Path
from waitress import serve
import os
import argparse
import webbrowser
import threading
import logging
import traceback
import apollo.server.schemas as schemas
import apollo.server.handlers as handlers
import apollo.server.pvlibw as pvlibw

logger = logging.getLogger(__name__)

class Config():
    """Stores directories and configuration settings of the Apollo Flask server. 
    
    Arguments:
        TIMESTAMP (str): The string used for timestamps in databases (default is ``TIMESTAMP``).
        HOME_DIR (pathlib.Path): The root directory of the server. 
        HTML_DIR (pathlib.Path): The public HTML directory to use. 
        DB_DIR (pathlib.Path): The root directory for databases used by the server. 
        DB_FILE (str): The default database file to use (in queries don't specify one).
        SQL_ROW_LIMIT (bool): The maximum number of rows to return from database queries. 
    """
    def __init__(self, home_dir=None, db_dir=None, db_file=None):
        if not home_dir:
            home_dir = os.path.realpath(__file__)
        if not db_dir:
            db_dir = os.path.realpath(__file__)
        if not db_file:
            db_file = os.path.realpath(__file__)
        self.TIMESTAMP =  "TIMESTAMP"
        self.HOME_DIR =   Path(home_dir)
        self.HTML_DIR =   self.HOME_DIR /"html"
        self.DB_DIR =    Path(db_dir)
        self.DB_FILE =    Path(self.DB_DIR/"default.db")
        self.SQL_ROW_LIMIT = 50000

cfg = Config()

##################################################################
# CONSTANTS
##################################################################

TIMESTAMP =  "TIMESTAMP"

app = Flask(__name__)

def _handle_bad_request(e):
    """If other request handlers fail, generate a response to the user. 
    """
    logger.error(str(e))
    traceback.print_exc()
    return 'Bad Request: '+str(e), 400

app.register_error_handler(400, _handle_bad_request)

#############################################
#############################################

def _solar_query():
    """Primary handler for queries to databases.

    The request encodes, as key-value paires the relevant information for a query. 
    A sample is given below. It will be converted to SQL and the results, formatted 
    as JSON, will be returned to the user. 
    
    .. code-block:: python   
    
        ImmutableMultiDict([
                ('source', 'solar_farm'), # database to use. 
                ('site', 'IRRADIANCE'),       # database table
                ('schema', 'solar_farm'),     # used to format results. 
                ('start', '1483160400000'),   # start timestamp in ms since 1/1970
                ('stop', '1483246800000'),    # end timestamp in ms since 1/1970
                ('attribute', 'UGAAPOA1IRR'), # table column
                ('attribute', 'UGAAPOA2IRR'), # table column
                ('attribute', 'UGAAPOA3IRR'), # table column
                ('groupby', 'yearmonthdayhourmin'), # SQL GROUP BY information
                ('statistic', 'AVG'),         # group statistic to retrieve
                ('statistic', 'MIN'),         # group statistic to retrieve
                ('statistic', 'MAX')])        # group statistic to retrieve
        
    See the ``apollo.server.handlers`` module documentation for more information. 

    Returns: 
        flask.Response:
            The response to the client. 
    """
    try:
        handler = handlers.factory(request, db_dir = cfg.DB_DIR, db_file=cfg.DB_FILE, row_limit=cfg.SQL_ROW_LIMIT)
        return handler.handle_request(request)
    except Exception as e:
        return _handle_bad_request(e)

#############################################
#############################################
@app.route("/status")
def _get_status():
    """Returns a simple status ``\{"status":1\}`` message to indicate that the server is alive. 

    Returns: 
        flask.Response:
            The response to the client. 
    """

    try:
        return jsonify({"status":1})
    except Exception as e:
        return _handle_bad_request(e)

#@app.route('/html/<path:path>')
def _send_files(path):
    """Retrieves a file from the HTML directory.
    
    Returns: 
        flask.Response:
            The response to the client. 
    """
    print(path)
    return send_from_directory(cfg.HTML_DIR, path)

@app.route('/pvlib')
def _get_pvlib():
    """Generates a JSON encoded forecast using PVLib-Python.
    
    The request should encode the ``start`` and ``stop`` times (as unix timestamps), 
    the latitude and longitude (as floats), and the PVLib model to run (as a string). 
    
    The results of the model forecast are returned formatted as JSON. 
    
    Returns: 
        flask.Response:
            The results of the forecast. 
    """
    try:
        
        args = request.args
        start = args.get("start",None)
        stop = args.get("stop",None)
        latitude = args.get("latitude",None)
        longitude = args.get("longitude",None)
        model = args.get("model",None)
        handler = pvlibw.PVLibForecastWrapper(model_name = model, 
                                              latitude = float(latitude), 
                                              longitude = float(longitude), 
                                              start = start, 
                                              stop = stop, 
                                              formatted=True)
        response_dictionary = handler.forecast()
        return jsonify(response_dictionary)
    except Exception as e:
        return _handle_bad_request(e)
         
@app.route('/sources')
def _get_sources():
    """Returns a JSON encoded list of data sources for use with the web client. 
    
    The list is stored in ``sources.json``, a file in the database directory. 
    
    Sample contents are given below. 

     .. code-block:: python
        
            [
            {"id":"solar_farm", "ext":".db", "label":"UGA Solar Farm", "schema":"solar_farm", "path": "solar_farm", "initial_start": "2017-01-01", "initial_stop": "2017-01-02"},
            {"id":"ALAPAHA", "ext":".db", "label":"Alapaha", "schema":"gaemn15min", "initial_start": "2013-01-01", "initial_stop": "2013-01-02"}
            ]    
        
    See the ``apollo.server.schemas`` module documentation for more information. 

    Returns: 
        flask.Response:
            The response to the client. 
    """
    try:
        with open(cfg.DB_DIR/"sources.json", 'r') as f:
            return f.read()        
    except Exception as e:
        return _handle_bad_request(e)

@app.route('/source-ui')
def _get_source_schemas():
    """Returns a json dictionary describing sources: 
        
    See the ``apollo.server.handlers`` module documentation for more information.         

    Returns: 
        flask.Response:
            The response to the client. 
    """
    try:
        schema = request.args.get("schema",None)
        source = request.args.get("source",None)
        result = schemas.get_source_ui_schema(cfg.DB_DIR,schema=schema,source=source)
        return jsonify(result)
    except Exception as e:
        return _handle_bad_request(e)
    

def _parsePath(inPath):
    return Path(inPath.replace('"','').replace("'",""))

def _config_from_args():
    parser = argparse.ArgumentParser(description="""Starts a Flask server to handle HTTP requests to the solar farm database. 
                                     EXAMPLE: 
>python -m apollo.server.solarserver --host 127.0.0.1 --port 5000 --html "I:\html" --dbdir "I:\db" --dbfile "solar_farm.db" --log DEBUG

                                     Static files are served from the specified HTML directory.""")
    parser.add_argument('--host', metavar='IP', type=str, dest='host',default='127.0.0.1',help='The IP to listen on. Default is 127.0.0.1.')
    parser.add_argument('--port', metavar='N', type=int, dest='port',default=5000,help='The port to listen on. Default is 5000.')
    parser.add_argument('--html', metavar='HTML_DIR', type=str, dest='html',default='',help='The directory for html and static files.')
    parser.add_argument('--dbdir', metavar='DB_DIR', type=str,dest='db_dir',default='',help='The directory storing the sqlite database(s) to use.')
    parser.add_argument('--dbfile', metavar='DB_FILE', type=str,dest='db_file',default='default.db',help='The default database file to use.')
    parser.add_argument('--dburl', metavar='dburl', type=str, dest='dburl',default='/apollo',help='The URL to bind to database queries.')
    parser.add_argument('--htmlurl', metavar='htmlurl', type=str, dest='htmlurl',default='/html',help='The URL to bind to static (html) queries.')
    parser.add_argument('--log', type=str, default='INFO', help='Sets the log level. One of INFO, DEBUG, ERROR, etc. Default is INFO.')
    
    args = parser.parse_args()

    logging.basicConfig(format='[{asctime}] {levelname}: {message}', style='{', level=args.log)
    logger.setLevel(args.log)

    if args.html == '':
        args.html = cfg.HTML_DIR
    else:
        cfg.HTML_DIR = _parsePath(args.html)


    if args.db_dir == '':
        args.db_dir = cfg.DB_DIR
    else:
        cfg.DB_DIR = _parsePath(args.db_dir)
        
    if args.db_file == '':
        args.db_file = cfg.DB_FILE
    else:
        cfg.DB_FILE = cfg.DB_DIR / args.db_file

    logging.info(" * host:"+str(args.host))
    logging.info(" * port:"+str(args.port))
    logging.info(" * html:"+str(args.html))
    logging.info(" * html url:"+str( args.htmlurl))
    logging.info(" * dbdir:"+str( args.db_dir))
    logging.info(" * dbfile:"+str( args.db_file))
    logging.info(" * db url:"+str( args.dburl))
    
    return args


if __name__ == '__main__':
    args = _config_from_args()
    try:
        threading.Timer(4, lambda: webbrowser.open("http://"+ args.host+":"+str(args.port)+"/html/index.html", new=2)).start()
    except:
        pass
    app.add_url_rule(args.dburl, '_solar_query', _solar_query)
    app.add_url_rule(args.htmlurl+"/<path:path>", '_send_files', _send_files)
    
    # Use when Flask is the primary server instead of Waitress. 
    #app.run(debug=False, host=args.host, port=args.port)
    
    # Use Waitress instead. Use host='0.0.0.0' to make it public. 
    serve(app, host=args.host, port=args.port)
