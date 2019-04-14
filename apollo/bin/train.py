import argparse

from apollo.models.base import save as save_model, list_known_models


def _is_abstract(cls):
    if not hasattr(cls, "__abstractmethods__"):
        return False  # an ordinary class
    elif len(cls.__abstractmethods__) == 0:
        return False  # a concrete implementation of an abstract class
    else:
        return True  # an abstract class


def parse_kwarg_list(kwarg_list):
    ''' Parses a list of kwargs formatted like "arg=val"
    '''
    return {pair[0]: pair[1] for pair
            in (kwarg.split('=') for kwarg in kwarg_list)}


def main(argv=None):
    models = {
        model.__name__: model
        for model in list_known_models()
        if not _is_abstract(model)
    }

    parser = argparse.ArgumentParser(
        description='Apollo Model Trainer',
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument(
        'model',
        type=str,
        choices=list(models.keys()),
        help='the type of the model to train',
    )

    parser.add_argument(
        '--start',
        '-b',
        default='2017-01-01T00:00',
        type=str,
        help='the first reftime in the training dataset',
    )

    parser.add_argument(
        '--stop',
        '-e',
        default='2017-12-31T18:00',
        type=str,
        help='the final reftime in the training dataset',
    )

    parser.add_argument(
        '--kwarg',
        type=str,
        action='append',
        help='a hyper-parameter for the model, e.g. `--kwarg arg1=val1`',
    )

    # parse args
    args = parser.parse_args(argv)
    args = vars(args)

    model_classname = args['model']
    ModelClass = models[model_classname]

    kwarg_list = args['kwarg'] if 'kwarg' in args else list()
    kwargs = parse_kwarg_list(kwarg_list)

    model = ModelClass(**kwargs)
    model.fit(first=args['start'], last=args['stop'])
    save_model(model)
    print(f'Model saved under name "{model.name}"')


if __name__ == '__main__':
    main()
