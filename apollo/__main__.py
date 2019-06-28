'''The main Apollo entrypoint.

This is a toolbox API. It is invoked as the `apollo` command followed by a
subcommand. Each subcommand is defined as a module in `apollo.bin`.
'''

import sys
import apollo.bin
apollo.bin.main(sys.argv[1:])
