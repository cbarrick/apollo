"""Defines routines for retrieving or creating information (schemas) describing 
the databases used by the Apollo web-front end. 

The data explorer of Apollo allows users to explore historical data stored in relational databases. 
Presenting the data to the user involves the use of human readable labels and other descriptive information, and these  
are stored in JSON files (i.e., `schema files`) associated with each database. 

For each database, there will be a single schema file. These are stored in a 
subdirectory of *db_home*, where *db_home* is the primary database directory (set at runtime). Separately, the
file ``sources.json``, contained in *db_home*, defines a list of the databases and their locations. 

Sample contents of ``sources.json`` are given below. 

 .. code-block:: python
    
    [
    {"id":"solar_farm", "ext":".db", "label":"UGA Solar Farm", "schema":"solar_farm", "initial_start": "2017-01-01", "initial_stop": "2017-01-02"},
    {"id":"ALAPAHA", "ext":".db", "label":"Alapaha", "schema":"gaemn15min", "initial_start": "2013-01-01", "initial_stop": "2013-01-02"}
    ]
    
Above, two databases are defined, ``solar_farm.db`` and ``ALAPAHA.db``. 

    1. ``id``: The name (file name) of the database (excluding the `.db` file extension). 
    2. ``ext``: The file extension of the databae (`.db`).
    3. ``label``: A human readable name of the database. 
    4. ``schema``: The name of the schema file, (excluding the ``.json`` extension) of the database. 
    5. ``initial_start``: The date to use as a default start date (in the web-interface). 
    6. ``initial_stop``: The date to use as a default stop date (in the web-interface). 

The directory structure corresponding to the above is: ::
    
    db_home
    ├── solar_farm
    │   └── solar_farm.db
    ├── gaemn15min
    │   └── ALAPAHA.db
    ├── sources.json
    ├── solar_farm.json
    └── gaemn15min.json

A schema file is used to formate results retrieved from a corresponding database. 

This module can be used to generate a schema file. There are two ways to do this. 
Either a set of manually created ``csv`` files describing the tables of the database is processed, or else the 
schema file is generated directly from the database itself. 

In the ``csv``-based method, each ``csv`` file corresponds to a table of the database in question, 
and each row of the file provides information on a single column of the table. 
The headers of the input ``csv`` files are assumed to be: 

 1. ``index``: indicates the column index of the attribute/column in the corresponding database table. 
 2. ``label``: indicates the column name of the attribute. 
 3. ``description``: a longer description of the attribute. 
 4. ``units``: the measurement units of the attribute. 
 5. ``sql_datatype``: the datatype in the database table (not currently used). 
 6. ``type``: the datatype to pass to the client side (web-browser) data renderer (Google Visualization API). 
 7. ``hidden``: TRUE or FALSE, indicating whether it should be hidden from the user or displayed.  

For instance, if  ``A.csv`` and ``B.csv`` are both in the directory ``/csv`` and contain the following 
text contents: ::

    index,label,description,units,sql_datatype,type,hidden
    1,TIMESTAMP,Unix integer timestamp,s,INTEGER,datetime,TRUE
    2,YEAR,Year of the observation,INTEGER,number,TRUE

then invoking 

>>> extract_csv_schema('/csv', outfile='default_schema.json', name="solar_farm", desc="UGA Solar Farm")

will generate the file ``default_schema.json``: 

.. code-block:: python

    {
      "name": "solar_farm",
      "description": "UGA Solar Farm",
      "tables": {
        "A": {
          "label": "Array A",
          "columns": {
            "TIMESTAMP": {
              "index": "1",
              "label": "TIMESTAMP",
              "description": "Unix integer timestamp",
              "units": "s",
              "sql_datatype": "INTEGER",
              "type": "datetime",
              "hidden": "TRUE"
            },
            "YEAR": {
              "index": "2",
              "label": "YEAR",
              "description": "Year of the observation",
              "units": "",
              "sql_datatype": "INTEGER",
              "type": "number",
              "hidden": "TRUE"
            }}},
        "B": {
          "label": "Array B",
          "columns": {
            "TIMESTAMP": {
              "index": "1",
              "label": "TIMESTAMP",
              "description": "Unix integer timestamp",
              "units": "s",
              "sql_datatype": "INTEGER",
              "type": "datetime",
              "hidden": "TRUE"
            },
            "YEAR": {
              "index": "2",
              "label": "YEAR",
              "description": "Year of the observation",
              "units": "",
              "sql_datatype": "INTEGER",
              "type": "number",
              "hidden": "TRUE"
            }}}
        }
    }

A similar dictionary can be generated directly from the database file using :func:`extract_db_schema`. However, 
in that case, the result must generally be edited to obtain labels that are fit 
to be read by humans. 

Once the schema files exist, information on specific attributes can be retrieved. 

>>>  get_schema_data('./db_home', 'solar_farm', 'IRRADIANCE', 'UGAAPOA1IRR')
{'index': '11',
 'label': 'UGAAPOA1IRR',
 'description': 'Pyranometer PYR01: Irradiance - Instantaneous - Plane of Array Value - from Logger',
 'units': 'w/m2',
 'sql_datatype': 'DOUBLE',
 'type': 'number',
 'hidden': 'FALSE'}

This information is generally used by the Apollo Data Explorer to present formatted
results to users. 

"""
import pandas as pd
import os
from pathlib import Path
import json
import logging
import sqlite3

logger = logging.getLogger(__name__)

NAME_KEY =   "name"
TABLES_KEY = "tables";
COL_KEY =    "columns"
INDEX_KEY =   "index"
LABEL_KEY =  "label";
DESCRIPTION_KEY = "description";
UNITS_KEY =  "units";
CHART_DATATYPE_KEY = "type";
HIDDEN_KEY =  "hidden";

# lazily store generated schema dictionaries (key indicates the schema)
schema_dict = {}

SOURCES = None

def extract_csv_schema(working_dir, outfile=None, name="", desc=""):
    """Process ``csv`` files in the given directory, returning a Python dictionary.
    
    It is assumed that each ``csv`` file defines information on a single table a single 
    datasource. Information from each file will be combined into a single dictionary
    for that source. 
    
    Arguments:
        working_dir (str): the directory containing the csv files to process. 
        outfile (str): The name of the generated schema json file. 
        name (str): The name to use for the data source. 
        desc (str): A longer description of the data source. 

    Returns:
        dict:
            A dictionary encoding the database schemaa. 
    """
    # 
    source_schema = {}
    source_schema[NAME_KEY] = name
    source_schema[DESCRIPTION_KEY] = desc
    
    schema_list = {}

    for root, dirs, files in os.walk(working_dir):
            for filename in files:
                if filename.endswith('.csv'):
                    name = filename.replace('.csv','')
                    schema = _extract_table_schema_from_csv(name, working_dir+filename)
                    schema_list[name] = schema
    source_schema[TABLES_KEY] = schema_list
    if outfile:
        with open(outfile, 'w') as outf:
            json.dump(source_schema, outf,indent=2)
    return source_schema

def _extract_table_schema_from_csv(name, filename):
    """Creates a dictionary from a ``csv`` file. Rows become entries in the dictionary. 
    
    Arguments:
        name (str): The name (e.g., table name) to associate with the csv file contents. 
        filename (str or pathlib.Path): The csv file to process. 

    Returns:
        dict:
            A dictionary encoding the csv file contents. 
    """
    df = pd.read_csv(filename, dtype=str, keep_default_na=False) 
    df1 = df.to_dict(orient='records')
    schema = {}
    schema[LABEL_KEY]  = name;
    columns = {}
    for entry in df1:
        key = entry[LABEL_KEY]
        temp_dict = {}
        for ekey in entry:
            temp_dict[ekey] = entry[ekey]
        columns[key] = temp_dict
    schema[COL_KEY] = columns
    return schema       



def extract_db_schema(file, outfile=None, name="", desc=""):
    """Generate a schema dictionary directly from a database file. Information on each 
    table in the database will be included. 
    
    Arguments:
        file (str): The name of the database file to access. 
        name (str): The name to use for the data source. 
        desc (str): A longer description of the data source. 
        outfile (str): An output file to write the schema to. 

    Returns:
        dict:
            A dictionary encoding the database schemaa. 
    """
    
    # 
    source_schema = {}
    source_schema[NAME_KEY] = name
    source_schema[DESCRIPTION_KEY] = desc
    
    schema_list = {}
    try:
        conn = sqlite3.connect(file)
        sql = "select name from sqlite_master where type = 'table'";
        c = conn.cursor()
        cur = c.execute(sql)
        tables = [];
        for t in cur.fetchall():
            tables.append(t[0])
        for table in tables:
            sql = f"PRAGMA table_info({table});"
            c = conn.cursor()
            cur = c.execute(sql)
            columns = cur.fetchall();
            schema = _process_db_table_columns(table, columns)
            schema_list[table] = schema
        conn.close()
        source_schema[TABLES_KEY] = schema_list
        if outfile:
            with open(outfile, 'w') as outf:
                json.dump(source_schema, outf,indent=2)
    except: 
        raise Exception(f"Problem extacting schema: {file}")
    return source_schema

def _process_db_table_columns(table, column_data):
    """Given a table of column data for a table, generate a dictionary 
    describing the column.

    .. code-block:: python
    
    "TIMESTAMP": {
              "index": "1",
              "label": "TIMESTAMP",
              "description": "Unix integer timestamp",
              "units": "s",
              "sql_datatype": "INTEGER",
              "type": "datetime",
              "hidden": "TRUE"
            },

   Arguments:
        table (str): The name of table 
        column_data (str): The results of ``PRAGMA table_info({table});``

    Returns:
        dict:
            A dictionary encoding metadata on the table. 
    """
    schema = {}
    schema[LABEL_KEY]  = table;
    columns = {}
    for column in column_data:
        temp_dict = {
                "index": column[0],
                "label": column[1],
                "description": column[1],
                "units": "",
                "sql_datatype": column[2],
                "type": "number",
                "hidden": "FALSE"
                }
        columns[column[1]] = temp_dict
    schema[COL_KEY] = columns
    return schema


def get_schema_data(schema_dir, schema, table, attribute):
    """Returns a dictionary of information on a column in a database table.
    
    The database schema is stored in a JSON file (`schema_dir/schema.json`). The 
    
    If the attribute (e.g., `Attr`) is not found, then a default dictionary is 
    generated and returned. The label and description are just the attribute 
    name provided, and it is assumed that the datatype is ``number``. 
        
    .. code-block:: python
      
        {'label': 'Attr','units': '','description': 'Attr','type': 'number'}        

   Arguments:
        schema_dir (str): The directory holding the schema file.  
        schema (str): The name (excluding ``.json``) of the schema file.
        table (str): The name of the relevant datbase table
        attribute (str): The name of the attribute/column to retrieve information on.

    Returns:
        dict:
            A dictionary encoding metadata on the specified table column. 
    """
    global schema_dict 
    if schema not in schema_dict:
        sfile = schema+".json"
        try:
            schema_dict[schema] = _load_json(Path(schema_dir)/sfile)
        except Exception as e:
            logger.error(f"Problem getting schema data for {schema}. {e} ")
            return None
    try:
        return schema_dict[schema][TABLES_KEY][table][COL_KEY][attribute]
    except Exception as e:
        logger.error(f"Problem getting schema data for {schema}. {e} ")
        metadata = {
                            LABEL_KEY:attribute,
                            UNITS_KEY:"",
                            DESCRIPTION_KEY:attribute,
                            CHART_DATATYPE_KEY:"number"}
        return metadata

def get_schema(schema_dir, schema):
    """Returns a dictionary encoding a data source schema.
    
    A global variable, ``schema_dict``, is used to store a dictionary of schemas 
    that have been loaded. If ``schema`` is defined as a key in that dictionary, then 
    the schema stored in the dictionary is returned. 
    
    If no entry is found, then the given directory is searched (if the schema is `sample`, then 
    `sample.json` is searched for). If found, it is read in as a Pythonb dictionary,
    stored in ``schema_dict`` and then returned. If not found, then ``None`` is returned. 

    Arguments:
        schema_dir (str): The directory to look in for the schema file. 
        schema (str): The name of the schema to look for. 

    Returns:
        dict:
            A dictionary encoding the schema. 
    """
    global schema_dict 
    try:
        if schema not in schema_dict:
            sfile = schema+".json"
            schema_dict[schema] = _load_json(Path(schema_dir)/sfile)
        return schema_dict[schema]
    except Exception as e:
        logger.error(f"Problem getting schema data for {schema}. {e} ")
        return None

def get_source_ui_schema(working_dir, schema=None,source=None):
    """Returns a dictionary encoding a data source schema, specifically for use in 
    the Apollo web-client. 
    
    This retrieves a schema and then formats the results for use in the drop-down 
    lists and other widgets of the web-client. 
    
    Arguments:
        working_dir (str): The directory to look in for the schema file (or database file). 
        schema (str): The name of the schema to look for. 
        source (str): The name of the data source (used as a key in the results). 
    Returns:
        dict:
            A dictionary encoding the schema. 
    """
    try:
        working_dir = Path(working_dir)
        json = None
        # if schema is "none", extract schema from db file. 
        if schema and schema == "none" and source:
            file = str(working_dir/(source+".db"))
            json = extract_db_schema(file,name="none", desc="none")
        # else look it up in a json file. 
        elif schema:
            json = get_schema(working_dir, schema)
        # if it's found, return a result dictionary encoding the schema.            
        if json:            
            result = {}
            result[NAME_KEY] = json[NAME_KEY] 
            result[DESCRIPTION_KEY] =  json[DESCRIPTION_KEY]
            result[TABLES_KEY] = [{"id":key, "label":json[TABLES_KEY][key][LABEL_KEY]} for key in json[TABLES_KEY]]
            result[COL_KEY] = [
                    [
                            {"id": key2, #json[schemas.TABLES_KEY][key][schemas.COL_KEY][key2][schemas.LABEL_KEY], 
                             "label": json[TABLES_KEY][key][COL_KEY][key2][DESCRIPTION_KEY]} 
                            for key2 in json[TABLES_KEY][key][COL_KEY] 
                            if json[TABLES_KEY][key][COL_KEY][key2][HIDDEN_KEY] == "FALSE"] 
                    for key in json[TABLES_KEY]]
            return result
    except Exception as e:
        logger.error(f"Problem getting schema data for {schema}. {e} ")
        return None
    
def get_sources(schema_dir, file):
    """Return a dictionary encoding a list of data sources. 
    
    This information stores the name and other information of the data sources 
    used by the Apollow web server. It is initially read from ``schema_dir/file`` but subsequently
    stored in a global variable (``SOURCES``). 

    Sample contents are given below. 

    .. code-block:: python
    
        [
        {"id":"solar_farm", "ext":".db", "label":"UGA Solar Farm", "schema":"solar_farm", "initial_start": "2017-01-01", "initial_stop": "2017-01-02"},
        {"id":"ALAPAHA", "ext":".db", "label":"Alapaha", "schema":"gaemn15min", "initial_start": "2013-01-01", "initial_stop": "2013-01-02"}
        ]
    
    1. ``id``: The name (file name) of the data source. 
    2. ``ext``: The file extension of the data source. 
    3. ``label``: A human readable name of the data source. 
    4. ``schema``: The name of the schema (file, excluding ``.json`` extension) of the data source. 
    5. ``initial_start``: The date to use as a default start date in the web-interface. 
    6. ``initial_stop``: The date to use as a default stop date in the web-interface. 
    
    """
    global SOURCES
    if not SOURCES:
        try:
            SOURCES = _load_json(Path(schema_dir)/file)
        except Exception as e:
            logger.error(str(e))
    return SOURCES
   
def _load_json(schema_file):
    """
    Loads a json file as a python dictionary.. 
    """
    results = None
    with open(schema_file, 'r') as f:
        results = json.load(f)
    return results

"""
if __name__ == "__main__":

    working_dir = "I:/Solar Radition Project Data April 2018/dist/schemas/raw_csv/uga/"
    outfile = working_dir + 'solar_farm.json'
    extract_csv_schema(working_dir, outfile, name="solar_farm", desc="UGA Solar Farm")
    
    working_dir = "I:/Solar Radition Project Data April 2018/dist/schemas/raw_csv/gaemn/"
    outfile = working_dir + 'gaemn15min.json'
    extract_csv_schema(working_dir, outfile, name="gaemn15min", desc="GAEMN 15 Minute Observations")
        
    '''
    working_dir = "I:/Solar Radition Project Data April 2018/dist/db/"
    outfile = working_dir + 'RAWGRIFFIN.db'
    x = extract_db_schema(outfile, name="griffin", desc="Griffin")
    print(x)
    '''

"""    
    
        
