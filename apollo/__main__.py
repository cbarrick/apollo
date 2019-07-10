'''The main Apollo entrypoint.

This is a toolbox API. It is invoked as the `apollo` command followed by a
subcommand. Each subcommand is defined as a module in `apollo.cli`.
'''

import sys
import apollo.cli
apollo.cli.main(sys.argv[1:])
