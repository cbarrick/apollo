import argparse
import logging
from pathlib import Path
import threading
from waitress import serve
import webbrowser

from apollo.server.solarserver import ServerConfig, setup_solar_server
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
        default=apollo.storage.get('assets/html'),
        help='The directory containing html and static files.'
    )

    parser.add_argument(
        '--dbdir',
        metavar='DB_DIR',
        type=str,
        default=apollo.storage.get('assets/db'),
        help='The directory storing the schema descriptions.'
    )

    parser.add_argument(
        '--dbfile',
        metavar='DB_FILE',
        type=str,
        default=apollo.storage.get('GA-POWER') / 'solar_farm.sqlite',
        help='The default database file to use.'
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

    html_directory = args.html
    db_directory = Path(str(args.dbdir).replace("'", '').replace('"', ''))
    db_filepath = args.dbfile

    logging.info('Server started with args:')
    for arg, val in vars(args).items():
        logging.info(f'  {arg}: {val}')

    cfg = ServerConfig(
        html_dir=html_directory,
        db_dir=db_directory,
        db_filepath=db_filepath)

    # open a webbrowser on the data explorer index page
    try:
        threading.Timer(4, lambda: webbrowser.open(
            "http://" + args.host + ":" + str(args.port) + "/html/index.html",
            new=2)).start()
    except:
        pass

    app = setup_solar_server(cfg)
    serve(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
