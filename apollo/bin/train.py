import argparse
from datetime import datetime

from apollo.models.base import save as save_model, list_known_models


def _is_abstract(cls):
    if not hasattr(cls, "__abstractmethods__"):
        return False  # an ordinary class
    elif len(cls.__abstractmethods__) == 0:
        return False  # a concrete implementation of an abstract class
    else:
        return True  # an abstract class


def main():
    MODELS = {model.__name__: model for model in list_known_models() if not _is_abstract(model)}

    parser = argparse.ArgumentParser(
        description='Apollo Machine Learning Model Trainer',
        argument_default=argparse.SUPPRESS,
    )
    # specify the type of model and give it a name
    parser.add_argument('model', type=str, choices=list(MODELS.keys()),
                        help='The type of the model that you would like to train.')

    parser.add_argument('--name', type=str, help='Human-readable name for the model.')

    # specify the training period
    parser.add_argument('--start', '-b', default='2017-01-01 00:00', type=str,
                       help='The first reftime in the dataset to be used for training.  '
                            'Any string accepted by numpy\'s datetime64 constructor will work.')

    parser.add_argument('--stop', '-e', default='2017-12-31 18:00', type=str,
                       help='The final reftime in the dataset to be used for training. '
                            'Any string accepted by numpy\'s datetime64 constructor will work.')

    # TODO: it would be nice if the user could also specify data kwargs and/or estimator hyperparameters

    # parse args
    args = parser.parse_args()
    args = vars(args)

    model_classname = args['model']
    ModelClass = MODELS[model_classname]

    current_dt = datetime.utcnow()
    model_name = args['name'] if 'name' in args \
        else f'{ModelClass.__name__}_{current_dt.year}-{current_dt.month}-{current_dt.day}'

    model = ModelClass(name=model_name)
    model.fit(first=args['start'], last=args['stop'])
    save_model(model)
    print(f'Model saved under name "{model_name}"')


if __name__ == '__main__':
    main()
