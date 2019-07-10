def description():
    import textwrap
    return textwrap.dedent('''\
    List items within the Apollo database.

    components:
        all        List everything in the database.
        models     List trained models.
        templates  List model templates.
    ''')


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description=description(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        'component',
        metavar='COMPONENT',
        choices=['all', 'models', 'templates'],
        default='all',
        nargs='?',
        help='The component to list, defaults to \'all\''
    )

    return parser.parse_args(argv)


def print_models(prefix=''):
    '''Print a listing of the models.
    '''
    from apollo import models
    for m in models.list_models():
        print(f'{prefix}{m}')


def print_templates(prefix=''):
    '''Print a listing of the templates.
    '''
    from apollo import models
    for t in models.list_templates():
        print(f'{prefix}{t}')


def print_all():
    '''Print all listings.
    '''
    print_models('models/')
    print_templates('templates/')


def main(argv):
    import sys

    args = parse_args(argv)

    if args.component == 'all':
        print_all()
    elif args.component == 'templates':
        print_templates()
    elif args.component == 'models':
        print_models()
    else:
        print(f'unknown component: {args.component}')
        sys.exit(2)

    sys.exit(0)
