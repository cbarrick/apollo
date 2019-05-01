''' Tools for running a :mod:`flask` server to respond to HTTP requests.

This is the primary server for handling queries through the Apollo Data Explorer 
web interface.  Both queries for static files and queries to the solar farm 
database (and other defined databases) are handled. 

When this script is invoked, directories for databases, HTML files, 
and supporting files should be set up correctly, and the script should be 
invoked with corresponding command line arguments.

TODO: much of this likely needs to be updated

A sample directory configuration is shown below. ::

    $APOLLO_DATA/assets
    ├──db
    │   ├── solar_farm_irr
    │   │   └── solar_farm_1min_irr.db
    │   ├── gaemn15min
    │   │   ├── ATTAPUL.db
    │   │   └── BLAIRSVI.db
    │   ├── sources.json
    │   ├── solar_farm.json
    │   ├── solar_farm_irr.json
    │   └── gaemn15min.json
    │   
    └──html
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
In this case, ``solar_farm_1min_irr.db``, ``ATTAPUL.db``, and ``BLAIRSVI.db``
are databases storting historical weather observations.
The file ``sources.json`` contains a list of all of the databases,
and the Data Explorer uses this list to determine which databases exist and
how to access them.

The format for ``sources.json`` is shown below. Each entry records the name 
(``id``) of the database file, its file extension, the name of its corresponding 
*schema* file, and the starting and ending dates for a sample query (the dates 
are used in the web interface).

 .. code-block:: python

    [
    {"id":"solar_farm", "ext":".db", "label":"UGA Solar Farm", "schema":"solar_farm", "initial_start": "2017-01-01", "initial_stop": "2017-01-02"},
    {"id":"ATTAPUL", "ext":".db", "label":"Attapulgus", "schema":"gaemn15min", "initial_start": "2013-01-01", "initial_stop": "2013-01-02"},
    ]

Files ``solar_farm_1min_irr.json`` and ``gaemn15min.json`` are schema files.
These store information on the columns in database tables
(their names, brief descriptions, their measurement units, etc.)
and this information is used to format the results of queries.
Each database should be associated with exactly one schema file.
The database should be stored in a subdirectory with the same name as the schema.
See :class:`apollo.server.schemas` for more information.

'''
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import logging
import traceback
import apollo.server.schemas as schemas
import apollo.server.handlers as handlers
import apollo.server.pvlibw as pvlibw

logger = logging.getLogger(__name__)


class ServerConfig(object):
    def __init__(self, html_dir, db_dir, db_filepath, sql_row_limit=50000):
        ''' Represents configuration settings for the Apollo data explorer app

        Args:
            html_dir (str):
                directory where raw html & css resources are located
            db_dir (str):
                directory containing json files describing the data sources
            db_filepath (str):
                path to the solar farm database file
        '''
        self.TIMESTAMP = "TIMESTAMP"  # TODO: not sure what this does
        self.HTML_DIR = Path(html_dir)
        self.DB_DIR = Path(db_dir)
        self.DB_FILE = Path(db_filepath)
        self.SQL_ROW_LIMIT = sql_row_limit


def setup_solar_server(cfg):
    ''' Creates a Flask server and registers routes

    Args:
        cfg (SolarConfig):
            The configuration settings for the server.

    Returns:
        flask.Flask: The flask server instance

    '''
    app = Flask(__name__)

    def _handle_bad_request(e):
        ''' Generates a 400 ERROR response if the other handlers fail '''
        logger.error(str(e))
        traceback.print_exc()
        return 'Bad Request: ' + str(e), 400

    def _solar_query():
        '''Primary handler for queries to databases.

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
        '''
        try:
            handler = handlers.factory(
                request,
                db_dir=cfg.DB_DIR,
                db_file=cfg.DB_FILE,
                row_limit=cfg.SQL_ROW_LIMIT)
            return handler.handle_request(request)
        except Exception as e:
            return _handle_bad_request(e)

    def _send_files(path):
        """Retrieves a file from the HTML directory.

        Returns:
            flask.Response:
                The response to the client.
        """
        return send_from_directory(cfg.HTML_DIR, path)

    @app.route('/status')
    def _get_status():
        '''Returns a simple status message indicating that the server is alive.

        Returns:
            flask.Response:
                The response to the client.
        '''

        try:
            return jsonify({"status": 1})
        except Exception as e:
            return _handle_bad_request(e)

    @app.route('/pvlib')
    def _get_pvlib():
        '''Generates a JSON encoded forecast using PVLib-Python.

        The request should encode the ``start`` and ``stop`` times (as unix timestamps),
        the latitude and longitude (as floats), and the PVLib model to run (as a string).

        The results of the model forecast are returned formatted as JSON.

        Returns:
            flask.Response:
                The results of the forecast.
        '''
        try:

            args = request.args
            start = args.get("start", None)
            stop = args.get("stop", None)
            latitude = args.get("latitude", None)
            longitude = args.get("longitude", None)
            model = args.get("model", None)
            handler = pvlibw.PVLibForecastWrapper(
                model_name=model,
                latitude=float(latitude),
                longitude=float(longitude),
                start=start,
                stop=stop,
                formatted=True)
            response_dictionary = handler.forecast()
            return jsonify(response_dictionary)
        except Exception as e:
            return _handle_bad_request(e)

    @app.route('/sources')
    def _get_sources():
        """Returns a JSON encoded list of data sources for use with the web client.

        The list is stored in a file ``sources.json`` in the database directory

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
            with open(cfg.DB_DIR / 'sources.json', 'r') as f:
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
            schema = request.args.get("schema", None)
            source = request.args.get("source", None)
            result = schemas.get_source_ui_schema(
                cfg.DB_DIR,
                schema=schema,
                source=source)
            return jsonify(result)
        except Exception as e:
            return _handle_bad_request(e)

    app.register_error_handler(400, _handle_bad_request)
    # TODO: might be able to use the usual @app.route registration method
    app.add_url_rule('/apollo', '_solar_query', _solar_query)
    app.add_url_rule('/html/<path:path>', '_send_files', _send_files)

    return app
