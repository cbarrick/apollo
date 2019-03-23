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


def main():
    models = {
        model.__name__: model
        for model in list_known_models()
        if not _is_abstract(model)
    }

    parser = argparse.ArgumentParser(
        description='Apollo Model Trainer',
        argument_default=argparse.SUPPRESS,
    )
    # specify the type of model and give it a name
    parser.add_argument('model', type=str, choices=list(models.keys()),
                        help='The type of the model to train.')

    # specify the training period
    parser.add_argument('--start', '-b', default='2017-01-01 00:00', type=str,
                        help='The first reftime in the training dataset. '
                             'Any string accepted by pandas\'s Timestamp '
                             'constructor will work.')

    parser.add_argument('--stop', '-e', default='2017-12-31 18:00', type=str,
                        help='The final reftime in the training dataset. '
                             'Any string accepted by pandas\'s Timestamp '
                             'constructor will work.')

    parser.add_argument('--kwarg', type=str, action='append',
                        help='Keyword arguments to pass to the model.'
                             'Should be formatted like '
                             '"--kwarg arg1=val1 --kwarg arg2=val2 . . ."')

    # parse args
    args = parser.parse_args()
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
