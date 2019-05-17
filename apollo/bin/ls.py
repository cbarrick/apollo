import argparse

from apollo.models.base import list_known_models, list_trained_models


def _is_concrete(cls):
    if not hasattr(cls, "__abstractmethods__"):
        return True  # an ordinary class
    elif len(cls.__abstractmethods__) == 0:
        return True  # a concrete implementation of an abstract class
    else:
        return False  # an abstract class


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='List Apollo models',
    )
    parser.add_argument(
        '--mode',
        '-m',
        type=str,
        choices=('saved', 'classes'),
        default='saved',
        help='If `saved`, list the names of previously trained models.'
             'If `classes`, list the names of subclasses of Model.'
    )
    parser.add_argument(
        '--concrete',
        '-c',
        action='store_true',
        help='Only print abstract subclasses of Model'
    )

    args = parser.parse_args(argv)

    if args.mode == 'classes':
        subclasses = list_known_models()
        if args.concrete:
            subclasses = filter(_is_concrete, subclasses)
        output = '\n'.join([m.__name__ for m in subclasses])
    else:
        output = '\n'.join(list_trained_models())

    print(output)


if __name__ == '__main__':
    main()
