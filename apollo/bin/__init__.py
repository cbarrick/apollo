import logging

from apollo.bin import _cli


def setup_logging(args):
    '''Setup the logging for the app.

    We use module-level loggers everywhere. Most of them are already
    initialized before we get to configure them. So we must reset the
    default log handlers that have already been created.
    '''
    if args.quiet:
        level = 'WARN'
    elif args.debug:
        level = 'DEBUG'
    else:
        level = args.log

    # Configure the root logger.
    log_format = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(log_format)
    root_logger = logging.getLogger('')
    root_logger.handlers = [log_handler]  # Replace the existing log handlers.

    # Only set the level for loggers under the `apollo` namespace.
    apollo_logger = logging.getLogger('apollo')
    apollo_logger.setLevel(level)


def main(argv):
    parser = _cli.subcommand_parser(
        description='the Apollo irradiance forecast system',
    )

    log_options = parser.add_mutually_exclusive_group()

    log_options.add_argument(
        '--quiet',
        action='store_true',
        help='only log error and warning messages (i.e. --log=WARN)'
    )

    log_options.add_argument(
        '--debug',
        action='store_true',
        help='log debug messages (i.e. --log=DEBUG)'
    )

    log_options.add_argument(
        '--log',
        metavar='LEVEL',
        type=str,
        default='INFO',
        help='the log level (default: INFO)'
    )

    args = parser.parse_args(argv)

    setup_logging(args)

    _cli.execute_subcommand(args)
