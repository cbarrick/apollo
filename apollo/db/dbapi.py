# -*- coding: utf-8 -*-
"""Defines database access routines.

This module serves as an interface to the database holding data 
logged from the solar farm and other weather stations. It is used the by the 
Apollo Data Explorer web server. The routines here, however, are general enough
that they could be used to access any :mod:`sqlite3`  database. 

The routines allow users to open and close connections, retrieve table and 
column information, execute SQL statements, and copy data from one table to 
another. The routines are implemented for use with :mod:`sqlite3` but could 
be modified to allow access to other relational database libraries. 

In a typical scenario, a :class:`.DBHandler` object is created, associated with a 
specific databse file. A connection to the file is opened using :func:`DBHandler.connect`, 
SQL statements are executed (using :func:`DBHandler.execute` or :func:`DBHandler.executescript`), 
and the connection is then closed using :func:`DBHandler.close`. Other functions
allow a list of tables or columns in tables to be quickly retrieved. 


Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

>>> dbh = DBHandler('C:/db/solar_farm.db')
>>> dbh.connect()
   Out[31]: <sqlite3.Connection at 0x21e11f0a8f0>

>>> print(dbh.tables())
  ['BASE', 'A', 'B', 'C', 'D', 'E', 'IRRADIANCE', 'TRACKING', 'BASE_TEMP', 'A_TEMP', 'B_TEMP', 'C_TEMP', 'D_TEMP', 'E_TEMP', 'IRRADIANCE_TEMP', 'TRACKING_TEMP']

>>> for column in dbh.columns('IRRADIANCE'):
>>>    print(column)
    (0, 'TIMESTAMP', 'INTEGER', 1, None, 1)
    (1, 'YEAR', 'INTEGER', 1, None, 0)
    (2, 'MONTH', 'INTEGER', 1, None, 0)
    (3, 'DAY', 'INTEGER', 1, None, 0)
    (4, 'HOUR', 'INTEGER', 1, None, 0)
    (5, 'MINUTE', 'INTEGER', 1, None, 0)
    (6, 'DAYOFYEAR', 'INTEGER', 0, None, 0)
    (7, 'CODE1', 'REAL', 0, None, 0)
    (8, 'CODE2', 'REAL', 0, None, 0)
    (9, 'CODE3', 'REAL', 0, None, 0)
    (10, 'UGAAPOA1IRR', 'REAL', 0, None, 0)
    (11, 'UGAAPOA2IRR', 'REAL', 0, None, 0)
    (12, 'UGAAPOA3IRR', 'REAL', 0, None, 0)
    (13, 'UGAAPOAREFIRR', 'REAL', 0, None, 0)
    (14, 'UGABPOA1IRR', 'REAL', 0, None, 0)
    (15, 'UGABPOA2IRR', 'REAL', 0, None, 0)
    (16, 'UGABPOA3IRR', 'REAL', 0, None, 0)
    (17, 'UGABPOAREFIRR', 'REAL', 0, None, 0)
    (18, 'UGADPOA1IRR', 'REAL', 0, None, 0)
    (19, 'UGADPOA2IRR', 'REAL', 0, None, 0)
    (20, 'UGADPOA3IRR', 'REAL', 0, None, 0)
    (21, 'UGADPOAREFIRR', 'REAL', 0, None, 0)
    (22, 'UGACPOA1IRR', 'REAL', 0, None, 0)
    (23, 'UGACPOA2IRR', 'REAL', 0, None, 0)
    (24, 'UGACPOA3IRR', 'REAL', 0, None, 0)
    (25, 'UGACPOAREFIRR', 'REAL', 0, None, 0)
    (26, 'UGAEPOA1IRR', 'REAL', 0, None, 0)
    (27, 'UGAEPOA2IRR', 'REAL', 0, None, 0)
    (28, 'UGAEPOA3IRR', 'REAL', 0, None, 0)
    (29, 'UGAEPOAREFIRR', 'REAL', 0, None, 0)
    (30, 'UGAMET01POA1IRR', 'REAL', 0, None, 0)
    (31, 'UGAMET01POA2IRR', 'REAL', 0, None, 0)
    (32, 'UGAMET02GHIIRR', 'REAL', 0, None, 0)
    (33, 'UGAMET02DHIIRR', 'REAL', 0, None, 0)
    (34, 'UGAMET02DNIIRR', 'REAL', 0, None, 0)
    (35, 'UGAMET02FIRIRR', 'REAL', 0, None, 0)

>>> for row in dbh.execute('select distinct YEAR from IRRADIANCE;'):
>>>    print(row)
    (2016,)
    (2017,)
    (2018,)
    (2019,)

>>> dbh.close()

""" 
import sqlite3
import logging
import pandas as pd

from pathlib import Path

logger = logging.getLogger(__name__)

    
def query_db(dbfile,sql):
    """Query a database, returning the results as a :class:`pandas.DataFrame`. 
    
    Internally, :func:`pandas.read_sql_query` is called. 
    
    Arguments:
        dbfile (str): The database file to access. 
        sql (str): The sql statement to execute. 
    
    Returns: 
        :class:`pandas.DataFrame`:
            The results of the database query. 
    """
    logger.debug("querying database..." + str(dbfile))
    logger.debug(sql)
    df = None
    try:
        conn = sqlite3.connect(dbfile)
        df = pd.read_sql_query(sql, conn)
        logger.debug("done querying database.")
    except Exception as e:
        logger.error(e)
    finally:
        if conn:
            conn.close()
    return df


class DBHandler:
    """Provides access to and insertion of data into SQLite datbases. 

    :class:`.DBHandler` objects store a path to the database file to access. 
    Once a connection is opened, a reference to it is also stored by the handler. 
    
    Attributes:
        db_file (str or pathlib.Path): The path to the database file.
        conn (sqlite3.Connection): A handle to a database connection (or ``None``).
    """
    
    def __init__(self,dbfile):
        """Create an instance of the handler, storing the database file name.
        
        ``self.conn`` is initialized to ``None``. 
        
        Arguments:
            dbfile (str or pathlib.Path): The path to the sqlite database that the handler should use. 
        
        """
        self.db_file = Path(dbfile)
        self.conn = None

    def connect(self):
        """Connect to the database, returning a reference to the connection. 
        
        Exceptions are suppressed. If an exception is 
        encountered, ``None`` is returned. 
        
        Returns:
            :class:`sqlite3.Connection`:
                A handle to the database connection. 
        """
        try:
            if self.conn:
                self.close()
            self.conn = sqlite3.connect(str(self.db_file))
            return self.conn
        except Exception as e:
            logger.error(f'Error connecting to db: {self.db_file}. {e}')
            return None

    def close(self):
        """Closes an open connection if there is one. 
        
        Before closing, :meth:`sqlite3.Connection.commit` is called. Afterwards, ``self.conn`` is set to ``None``.
        If there is no open connection, then the method does nothing. 
        """
        if self.conn is not None:
            try:
                self.conn.commit()
                self.conn.close()
                self.conn = None
            except Exception as e:
                logger.error(f'Error closing connection. {e}')
                
    def execute(self, sql, commit=False):
        """Executes the SQL statement, returning the resulting cursor.
        
        If ``commit`` is true, then the transaction commits if the execution 
        was successful and and rolls back if it fails. 
        
        Arguments:
            sql (str): The statement to execute.
            commit (bool): Peform a commit after execution. 
        
        Returns:
            :class:`sqlite3.Cursor`: 
                A reference to the cursor. 

        """
        try:
            if commit:
                    with self.conn:
                         return self.conn.execute(sql)            
            return self.conn.execute(sql)            
        except Exception as e:
            logger.error(f'Error executing statement: {sql}. {e}')

    def executescript(self,sql, commit=False):
        """Executes the SQL statement(s), returning the resulting cursor.
        
        If ``commit`` is true, then the transaction commits if the execution 
        was successful and and rolls back if it fails. 
        
        This method allows multiple SQL statements to be encoded in ``sql``. 
            
        Arguments:
            sql (str): The script to execute.
            commit (bool): Peform a commit after execution. 

        Returns:
            :class:`sqlite3.Cursor`: 
                A reference to the cursor. 
        """
        try:
            if commit:
                with self.conn:
                     return self.conn.executescript(sql)            
            return self.conn.executescript(sql)            
        except Exception as e:
            logger.error(f'Error executing statement: {sql}. {e}')

    def tables(self):
        """Returns the names of the tables in the database.

        An open database connection should exist before invoking this method. 
        
        Returns:
            list: 
                A list of table names.
        """
        return [t[0] for t in self.execute("select name from sqlite_master where type = 'table'")]

    def columns(self, table):
        """Returns a list of entries with information on columns.
        
        This is executes ``PRAGMA table_info(table)``, returning the results. 

        An open database connection should exist before invoking this method. 
        
        Arguments:
            table (str): The name of the table to examine. 
        Returns:
            list: 
                A list containing information on the table columns. 
        """
        return self.execute(f"PRAGMA table_info({table});").fetchall()

    def column_names(self, table):
        """Returns the names of columns in the given table.
        
        An open database connection should exist before invoking this method. 

        Arguments:
            table (str): The name of the table to examine. 
        Returns:
            ``list``: a list of table column names. 
        """
        return [row[1] for row in self.columns(table)]

    def copy_table(self, source,target):
        """Copies one table into another. 
        
        Old records in the target are replaced if there is a conflict.
        
        An open database connection should exist before invoking this method. 
        
        Arguments:
            source (str): The name of the table to copy from. 
            target (str): The name of the table to copy into.
        """
        statement = f"INSERT OR REPLACE INTO {target} SELECT * FROM {source}"
        self.execute(statement, commit=True)

    def insert_dataframe(self, df,table):
        """Inserts a :class:`pandas.DataFrame` into a table. 
        
        Uses :func:`pandas.DataFrame.to_sql`. The dataframe must match
        the format of the table. 
        If duplicates keys are found, then preexisting values are overwritten. 
        
        An open database connection should exist before invoking this method. 
       
        Arguments:
            df (:class:`pandas.DataFrame`): The dataframe to insert. 
            table (str): The name of the table to insert into.
       
        """
        df.to_sql(table, self.conn, if_exists='replace', index=False)
        