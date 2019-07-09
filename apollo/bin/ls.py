import argparse
import sys

from apollo import models


def main(argv):
    parser = argparse.ArgumentParser(
        description='list items within the Apollo database'
    )

    parser.add_argument(
        'component',
        choices=['models', 'templates'],
        help='the component of the Apollo database to list'
    )

    args = parser.parse_args(argv)

    if args.component == 'templates':
        items = models.list_templates()
    elif args.component == 'models':
        items = models.list_models()
    else:
        print(f'unknown component: {args.component}')
        sys.exit(2)

    if len(items) != 0:
        print(*items, sep='\n')
    sys.exit(0)
