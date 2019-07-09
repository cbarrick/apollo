import argparse
import sys

from apollo import models


def main(argv):
    parser = argparse.ArgumentParser(
        description='list trained models or model templates'
    )

    parser.add_argument(
        '-t',
        '--templates',
        action='store_true',
        help='list model templates instead of trained models'
    )

    args = parser.parse_args(argv)

    if args.templates:
        templates = models.list_templates()
        if len(templates) == 0:
            print('no model templates', file=sys.stderr)
            sys.exit(1)
        else:
            print(*templates, sep='\n')
            sys.exit(0)

    else:
        models = models.list_models()
        if len(models) == 0:
            print('no trained models', file=sys.stderr)
            sys.exit(1)
        else:
            print(*models, sep='\n')
            sys.exit(0)
