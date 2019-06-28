import logging

from apollo.bin import _cli


def setup_logging(level):
    '''Setup the logging for the app.

    We use module-level loggers everywhere. Most of them are already
    initialized before we get to configure them. So we must reset the
    default log handlers that have already been created.
    '''
    # Configure the root logger.
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter(
        '[{asctime}] {levelname}: {message}',
        style='{',
    ))
    root_logger = logging.getLogger('')
    root_logger.handlers = [log_handler]  # Replace the existing log handlers.

    # Configure all loggers under the `apollo` namespace.
    apollo_logger = logging.getLogger('apollo')
    apollo_logger.setLevel(level)


def main(argv):
    parser = _cli.subcommand_parser(
        description='the Apollo irradiance forecast system',
    )

    parser.add_argument(
        '-l',
        '--log',
        metavar='LEVEL',
        type=str,
        default='INFO',
        help='the log level (default: INFO)'
    )

    args = parser.parse_args(argv)

    setup_logging(args.log)

    _cli.execute_subcommand(args)
