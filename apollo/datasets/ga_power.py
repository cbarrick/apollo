'''Provides access to a subset of the Georgia Power target data.

The data must have been previously preprocessed into a CSV file named
``mb-007.{group}.log`` where ``{group}`` is an arbitrary identifier for the
data.
'''

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import xarray as xr
import sqlite3

from apollo import storage, timestamps


# Module level logger
logger = logging.getLogger(__name__)


def interval(year=None, month=None, day=None, hour=None, minute=None):
    '''Helper function to group a timeseries DataFrame by interval.

    Args:
        year (int): Group by rounding the year.
        month (int): Group by rounding the month.
        day (int): Group by rounding the day.
        hour (int): Group by rounding the hour.
        minute (int): Group by rounding the minute.

    Returns:
        Callable[[datetime.datetime or pandas.Timestamp], datetime.datetime]:
            A function that maps arbitrary datetimes to reference datetimes by
            rounding interval properties like `second` and `minute`. The input
            may be a :class:`datetime.datetime` from the standard library or a
            :class:`pandas.Timestamp`. The return value is a
            :class:`datetime.datetime`.

    Examples:
        Group a DataFrame into 15 minute blocks:
        >>> df.groupby(interval(minute=15))

    Warning:
        Getting the right arguments for the interval you want is subtle.
        Read the code to see exactly how this works.
    '''
    # Only one of the kwargs should be given. The rest are derived.
    # We use values that exceed the traditional range for the derived values.
    # This is to account for any number of "leap" issues. It has not been
    # demonstrated that this is needed, but it doesn't hurt the correctness.

    # If year is rounded, all 12 months should round down.
    if year is not None:
        assert month is None
        month = 13
    else:
        year = 1

    if month is not None:
        assert day is None
        day = 32
    else:
        month = 1

    if day is not None:
        assert hour is None
        hour = 25
    else:
        day = 1

    if hour is not None:
        assert minute is None
        minute = 70
    else:
        hour = 1

    if minute is None:
        minute = 1

    def round_down(num, divisor, origin=0):
        num -= origin
        num -= (num % divisor)
        num += origin
        return num

    def grouper(t):
        group_year = round_down(t.year, year, origin=1)
        group_month = round_down(t.month, month, origin=1)
        group_day = round_down(t.day, day, origin=1)
        group_hour = round_down(t.hour, hour)
        group_minute = round_down(t.minute, minute)
        return datetime(group_year, group_month, group_day, group_hour, group_minute,
                tzinfo=t.tzinfo)

    return grouper


def open_mb007(*cols, group=2017):
    '''Open a Georgia Power target file.

    The data must have been previously preprocessed into a CSV file named
    ``mb-007.{group}.log`` where ``{group}`` is an arbitrary identifier for the
    data.

    Arguments:
        *cols (str or int):
            The columns to read from the file. These may be names or indices.
            The reftime column (0) will always be read.
        group (Any):
            An identifier for the file to read.

    Returns:
        xarray.Dataset:
            The dataset giving the targets.
    '''
    # The data directory contains more than just the mb-007 labels.
    data_dir = storage.get('GA-POWER')
    path = data_dir / f'mb-007.{group}.log'

    # Ensure reftime is always selected and is a list.
    if 'reftime' not in cols and 0 not in cols:
        cols = ['reftime', *cols]
    else:
        cols = list(cols)

    # Load the CSV and aggregate by hour.
    df = pd.read_csv(path, usecols=cols, index_col='reftime')
    df = df.dropna()
    df.index = pd.to_datetime(df.index, infer_datetime_format=True)
    df = df.groupby(interval(hour=1)).mean()

    # For some reason, the index name isn't set by default.
    df.index.name = 'reftime'

    # In apollo, all datasets should be presented as xarray.
    df = df.to_xarray()

    return df


def open_sqlite(*cols, start, stop):
    ''' Open Georgia Power irradiance data from a sqlite database

    The data must be stored in a sqlite database named solar_farm.sqlite in the <APOLLO_DATA>/GA-POWER directory.
    The database should include a table named "IRRADIANCE" storing irradiance readings indexed by a integer column
    named "TIMESTAMP".

    The range defined by the `start` and `stop` arguments is inclusive.
    If data is not present for a portion of the requested range, then it will be silently excluded.


    Args:
        *cols (str):
            The column names to read from the database.
            The TIMESTAMP column will always be read.
        start (str or Timestamp:
            The Timestamp of the first reftime which should be read.
        stop:
            The Timestamp of the last reftime which will be read.

    Returns:

    '''
    data_dir = storage.get('GA-POWER')
    path = data_dir / 'solar_farm.sqlite'
    connection = sqlite3.connect(str(path))

    start, stop = timestamps.utc_timestamp(start), timestamps.utc_timestamp(stop)

    # convert start and stop timestamps to unix epoch in seconds
    unix_start = start.value // 10**9
    unix_stop = stop.value / 10**9

    # Ensure that `cols` is a list
    cols = list(cols)

    # build the query
    query = f'SELECT TIMESTAMP as reftime' + (', ' if len(cols) > 0 else '')
    query += ','.join(cols)
    query += ' FROM IRRADIANCE WHERE reftime BETWEEN ? AND ?;'
    params = [unix_start, unix_stop]

    # load data and aggregate by hour
    df = pd.read_sql_query(
        sql=query,
        con=connection,
        params=params,
        index_col='reftime',
        parse_dates=['reftime'])
    df = df.dropna()

    # clip into selected date range
    df = df[start:stop]

    # take one-hour averages
    if len(df.index) > 0:
        df = df.groupby(interval(hour=1)).mean()

    # most of apollo assumes the index name will be `reftime`
    df.index.name = 'reftime'

    # In apollo, all datasets should be presented as xarray
    df = df.to_xarray()
    return df


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

