import argparse
import logging
import pkg_resources
from pathlib import Path
import webbrowser
import threading

import apollo.storage

logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Starts a Flask server to handle HTTP requests to the '
                    'data explorer app.'
    )

    parser.add_argument(
        '--host',
        metavar='IP',
        type=str,
        default='127.0.0.1',
        help='The IP to listen on. Default is 127.0.0.1.'
    )

    parser.add_argument(
        '--port',
        metavar='N',
        type=int,
        default=5000,
        help='The port to listen on. Default is 5000.'
    )

    parser.add_argument(
        '--html',
        metavar='HTML_DIR',
        type=str,
        default='',
        help='The directory containing html and static files.'
    )

    parser.add_argument(
        '--dbdir',
        metavar='DB_DIR',
        type=str,
        default=apollo.storage.get('GA-POWER'),
        help='The directory storing the sqlite database(s) to use.'
    )

    parser.add_argument(
        '--dbfile',
        metavar='DB_FILE',
        type=str,
        default='solar_farm.sqlite',
        help='The default database file to use.'
    )

    parser.add_argument(
        '--dburl',
        metavar='dburl',
        type=str,
        default='/apollo',
        help='The URL to bind to database queries.'
    )

    parser.add_argument(
        '--log',
        type=str,
        default='INFO',
        choices=('INFO', 'DEBUG', 'WARN', 'ERROR'),
        help='Sets the log level. Default is INFO.'
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        format='[{asctime}] {levelname}: {message}',
        style='{',
        level=args.log
    )
    logger.setLevel(args.log)

    # TODO: see if we can get rid of this and just use pkg_resources to access static html assets
    html_directory = pkg_resources.resource_filename('apollo.assets.html', '/')
    print(html_directory)

    db_directory = Path(str(args.dbdir).replace("'", '').replace('"', ''))
    db_filepath = db_directory / args.dbfile

    logging.info('Server started with args:')
    for arg, val in vars(args).items():
        logging.info(f'  {arg}: {val}')

    # TODO: start server


if __name__ == '__main__':
    main()
