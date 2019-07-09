from apollo.bin import _cli

def main(argv):
    parser = _cli.subcommand_parser(
        description='subcommands for working with models',
    )

    args = parser.parse_args(argv)
    _cli.execute_subcommand(args)
