'''Provides access to a subset of the Georgia Power target data.

The data must have been previously preprocessed into a CSV file named
``mb-007.{group}.log`` where ``{group}`` is an arbitrary identifier for the
data.
'''

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import logging
import sqlite3

import pandas as pd
import xarray as xr

import apollo
from apollo import storage


logger = logging.getLogger(__name__)


def raw_connect():
    '''Get a connection to the SQLite database.

    This is a lower-level function to create a SQLite connection object. It is
    your responsibility to commit or roll back your transactions and to close
    the connection when you are done.

    To accomplish these tasks automatically with a context manager, use the
    higher-level :func:`connect` function.

    The database is located at ``$APOLLO_DATA/GA-POWER/solar_farm.sqlite``.

    Returns:
        sqlite3.Connection:
            The database connection.
    '''
    data_dir = storage.get('GA-POWER')
    path = data_dir / 'solar_farm.sqlite'
    return sqlite3.connect(str(path))


@contextmanager
def connect():
    '''Get a connection to the SQLite database.

    This should be used as a context manager in a ``with`` block. When you exit
    the block, the current commit will be automatically commited (or rolled
    back if an exception occured) and the connection will be closed.

    The database is located at ``$APOLLO_DATA/GA-POWER/solar_farm.sqlite``.

    Example:
        Use this as a context manager. You don't need to explicitly commit or
        close the connection::

            >>> with ga_power.connect() as con:
            ...     query = 'SELECT * FROM IRRADIANCE'
            ...     return pd.read_sql_query(query, con)
    '''
    # A sqlite3.Connection can be used as a context manager
    # that automatically commits or rolls back...
    with raw_connect() as con:
        yield con

    # ...but it doesn't automatically close.
    con.close()


def open(start='2016-01-01', stop='now'):
    '''Load Georgia Power irradiance data between two timestamps.

    Args:
        start (timestamp-like):
            The Timestamp of the first reftime which should be read. The
            default is January 1st, 2016. We have no earlier data.
        stop (timestamp-like):
            The Timestamp of the last reftime which will be read. The default
            is the current time when the function is called.

    Returns:
        pd.DataFrame:
            The contents of the query.
    '''
    start = apollo.Timestamp(start)
    stop = apollo.Timestamp(stop)

    with connect() as con:
        df = pd.read_sql_query(f'''
            SELECT * FROM IRRADIANCE
            WHERE TIMESTAMP BETWEEN {start.timestamp()} AND {stop.timestamp()}
            ''',
            con=con,
            index_col='TIMESTAMP',
            parse_dates=['TIMESTAMP'])

    df.index.name = 'time'
    df.index = df.index.tz_localize('UTC')
    return df


# ---------------------------------------------------------------------------
# TODO: Everything below this line is redundant. I believe `open` above
# and the xarray API are sufficient to handle our needs. BUT the server
# subpackage relies on these routines, so we can't remove them yet.


def query_db(dbfile, sql):
    ''' Query a database, returning the results as a :class:`pandas.DataFrame`.

    Arguments:
        dbfile (str):
            The sqlite database file to query.
        sql (str):
            The sql statement to execute.

    Returns:
        :class:`pandas.DataFrame`:
            The results of the database query.

    '''
    logger.debug("querying database..." + str(dbfile))
    logger.debug(sql)
    conn = None
    try:
        conn = sqlite3.connect(dbfile)
        df = pd.read_sql_query(sql, conn)
        logger.debug("done querying database.")
        return df
    except Exception as e:
        logger.error(e)
    finally:
        if conn is not None:
            conn.close()


class DBHandler:
    ''' Object-oriented interface for interacting with SQLite databases

    :class:`.DBHandler` objects store a path to the database file to access
    as well as a reference to an open connection to that database.

    Attributes:
        db_file (str or pathlib.Path):
            The path to the database file.
        conn (sqlite3.Connection):
            A handle to a database connection (or ``None``).
    '''

    def __init__(self, dbfile):
        ''' Initialize a :class:`.DBHandler`

        Args:
            dbfile (str or pathlib.Path):
                The path to the sqlite database that the handler will target.
        '''
        self.db_file = Path(dbfile)
        self.conn = None

    def connect(self):
        ''' Opens a connection to this handler's database

        Exceptions are suppressed. If an exception is caught,
        ``None`` is returned.

        Returns:
            :class:`sqlite3.Connection`:
                A handle to the database connection.
        '''
        try:
            if self.conn:
                self.close()
            self.conn = sqlite3.connect(str(self.db_file))
            return self.conn
        except Exception as e:
            logger.error(f'Error connecting to db: {self.db_file}. {e}')
            return None

    def close(self):
        ''' Close the connection maintained by this handler.

        Before closing, :meth:`sqlite3.Connection.commit` is called.
        If a connection has not been opened with :meth:`DBHandler.connect`,
        this method is a no-op

        Returns: None

        '''
        if self.conn is not None:
            try:
                self.conn.commit()
                self.conn.close()
                self.conn = None
            except Exception as e:
                logger.error(f'Error closing connection. {e}')

    def execute(self, sql, commit=False):
        ''' Execute a SQL statement against the database

        Args:
            sql (str):
                The statement to execute.
            commit (bool):
                Peform a commit after execution.

        Returns:
            :class:`sqlite3.Cursor`:
                A reference to the cursor with query results.

        '''
        try:
            if commit:
                with self.conn:
                    return self.conn.execute(sql)
            return self.conn.execute(sql)
        except Exception as e:
            logger.error(f'Error executing statement: {sql}. {e}')

    def executescript(self, sql, commit=False):
        ''' Executes a SQL script with one or more statements

        Args:
            sql (str):
                The script to execute.
            commit (bool):
                Peform a commit after execution.

        Returns:
            :class:`sqlite3.Cursor`:
                A reference to the cursor containing results.

        '''
        try:
            if commit:
                with self.conn:
                    return self.conn.executescript(sql)
            return self.conn.executescript(sql)
        except Exception as e:
            logger.error(f'Error executing statement: {sql}. {e}')

    def tables(self):
        ''' Lists the names of tables in the database

        An open database connection should exist before invoking this method.

        Returns:
            list:
                A list of table names.

        '''
        return [t[0] for t in self.execute(
            "select name from sqlite_master where type = 'table'")]

    def columns(self, table):
        ''' List column information

        This is executes ``PRAGMA table_info(table)``, returning the results.

        An open database connection should exist before invoking this method.

        Args:
            table (str):
                The name of the table to examine

        Returns:
            list:
                A list containing information on the table columns.
        '''
        return self.execute(f"PRAGMA table_info({table});").fetchall()

    def column_names(self, table):
        ''' List the names of columns in the given table

        Args:
            table (str):
                The name of the table to examine.

        Returns:
            list:
                A list of column names in the table.

        '''
        return [row[1] for row in self.columns(table)]

    def copy_table(self, source, target):
        ''' Copy one table into another

        Old records in the target are replaced if there is a conflict.

        An open database connection should exist before invoking this method.

        Args:
            source (str):
                The name of the table to copy from.
            target (str):
                The name of the table to copy into.

        Returns: None

        '''
        statement = f"INSERT OR REPLACE INTO {target} SELECT * FROM {source}"
        self.execute(statement, commit=True)

    def insert_dataframe(self, df, table):
        ''' Insert a :class:`pandas.DataFrame` into a table

        The dataframe must match the format of the table.
        If duplicates keys are found, then preexisting values are overwritten.

        An open database connection should exist before invoking this method.

        Args:
            df (:class:`pandas.DataFrame`):
                The dataframe to insert.
            table (str):
                The name of the table to insert into.

        Returns: None

        '''
        df.to_sql(table, self.conn, if_exists='replace', index=False)
