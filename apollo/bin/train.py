import argparse

from apollo.models.base import save as save_model, list_known_models


def _is_abstract(cls):
    if not hasattr(cls, "__abstractmethods__"):
        return False  # an ordinary class
    elif len(cls.__abstractmethods__) == 0:
        return False  # a concrete implementation of an abstract class
    else:
        return True  # an abstract class


def parse_kwarg_list(list):
    ''' Parses a list of kwargs formatted like "arg=val" '''
    args = dict()
    for pair in list:
        key, value = pair.split('=')
        args[key] = value

    return args


def main():
    MODELS = {model.__name__: model for model in list_known_models() if not _is_abstract(model)}

    parser = argparse.ArgumentParser(
        description='Apollo Machine Learning Model Trainer',
        argument_default=argparse.SUPPRESS,
    )
    # specify the type of model and give it a name
    parser.add_argument('model', type=str, choices=list(MODELS.keys()),
                        help='The type of the model that you would like to train.')

    # specify the training period
    parser.add_argument('--start', '-b', default='2017-01-01 00:00', type=str,
                       help='The first reftime in the dataset to be used for training.  '
                            'Any string accepted by numpy\'s datetime64 constructor will work.')

    parser.add_argument('--stop', '-e', default='2017-12-31 18:00', type=str,
                       help='The final reftime in the dataset to be used for training. '
                            'Any string accepted by numpy\'s datetime64 constructor will work.')

    parser.add_argument('--data_kwargs', type=str, nargs='*',
                        help='kwargs used to load data.  Should be formatted like `arg1=val1 arg2=val2 ...`')
    parser.add_argument('--model_kwargs', type=str, nargs='*',
                        help='kwargs used to set hyperparameters. Should be formatted like `arg1=val1 arg2=val2 ...`')
    parser.add_argument('--kwargs', type=str, nargs='*',
                        help='Other kwargs used for model initialization. '
                             'Should be formatted like `arg1=val1 arg2=val2 ...`')

    # parse args
    args = parser.parse_args()
    args = vars(args)

    model_classname = args['model']
    ModelClass = MODELS[model_classname]

    data_kwargs = parse_kwarg_list(args['data_kwargs']) if 'data_kwargs' in args else None
    model_kwargs = parse_kwarg_list(args['model_kwargs']) if 'model_kwargs' in args else None
    kwargs = parse_kwarg_list(args['kwargs']) if 'kwargs' in args else dict()

    model = ModelClass(data_kwargs=data_kwargs, model_kwargs=model_kwargs, **kwargs)
    model.fit(first=args['start'], last=args['stop'])
    save_model(model)
    print(f'Model saved under name "{model.name}"')


if __name__ == '__main__':
    main()
