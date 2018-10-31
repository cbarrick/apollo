import argparse
import logging
import numpy as np
import pandas as pd

from sklearn.metrics.regression import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import make_scorer

from apollo.prediction.SKPredictor import SKPredictor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PREDICTORS = dict()
for predictor_class in SKPredictor.__subclasses__():
    PREDICTORS[predictor_class.get_name()] = predictor_class

DEFAULT_METRICS = {
    'mse': make_scorer(mean_squared_error),
    'mae': make_scorer(mean_absolute_error),
    'r2': make_scorer(r2_score)
}


def main():
    parser = argparse.ArgumentParser(
        description='Apollo: Solar Radiation Prediction with Machine Learning',
        argument_default=argparse.SUPPRESS,
    )

    # arguments that are common across all sub-commands

    parser.add_argument('--model', '-m', default='dtree', type=str, choices=list(PREDICTORS.keys()),
                        help='The name of the model that you would like to run.')

    parser.add_argument('--target_hours', '-f', default=24, type=int,
                        help='Generate predictions for each our up to this hour. '
                             'Should be an integer between 1 and 36.')

    parser.add_argument('--target', '-t', default='UGA-C-POA-1-IRR', type=str,
                        help='The variable from GA_POWER to target.')

    subcommands = parser.add_subparsers()

    # train
    train = subcommands.add_parser('train', argument_default=argparse.SUPPRESS, description='Train a model.')
    train.set_defaults(action='train')

    train.add_argument('--start', '-b', default='2017-01-01 00:00', type=str,
                        help='The first reftime in the dataset to be used for training.  '
                             'Any string accepted by numpy\'s datetime64 constructor will work.')

    train.add_argument('--stop', '-e', default='2017-12-31 18:00', type=str,
                        help='The final reftime in the dataset to be used for training. '
                             'Any string accepted by numpy\'s datetime64 constructor will work.')

    train.add_argument('--no_tune', '-p', action='store_true',
                       help='If set, hyperparameter tuning will NOT be performed during training.')

    train.add_argument('--num_folds', '-n', default=3, type=int,
                       help='If `tune` is enabled, the number of folds to use during the cross-validated grid search. '
                            'Ignored if tuning is disabled.')

    # evaluate
    evaluate = subcommands.add_parser('evaluate', argument_default=argparse.SUPPRESS,
                                      description='Evaluate a model using n-fold cross-validation')
    evaluate.set_defaults(action='evaluate')

    evaluate.add_argument('--start', '-b', default='2017-05-01 00:00', type=str,
                        help='The first reftime in the dataset to be used for evaluation.  '
                             'Any string accepted by numpy\'s datetime64 constructor will work.')

    evaluate.add_argument('--stop', '-e', default='2017-05-30 18:00', type=str,
                        help='The final reftime in the dataset to be used for evaluation. '
                             'Any string accepted by numpy\'s datetime64 constructor will work.')

    evaluate.add_argument('--num_folds', '-n', default=3, type=int,
                          help='The number of folds to use when computing cross-validated accuracy.')

    # predict
    predict = subcommands.add_parser('predict', argument_default=argparse.SUPPRESS,
                                     description='Make predictions using a trained model.')
    predict.set_defaults(action='predict')

    predict.add_argument('--reftime', '-r', default='2018-01-01 00:00', type=str,
                        help='The reftime for which predictions should be made.  '
                             'Any string accepted by numpy\'s datetime64 constructor will work.  '
                             'Ignored if the `latest` flag is set.')

    predict.add_argument('--latest', '-l', action='store_true',
                       help='If set, a prediction will be generated for the past reftime which is closest to the '
                            'current datetime.')

    predict.add_argument('--summary_dir', '-z', default='./summaries', type=str,
                         help='The directory where summary files will be written.')

    predict.add_argument('--output_dir', '-o', default='./predictions', type=str,
                         help='The directory where predictions will be written.')

    # parse args and invoke the correct model
    args = parser.parse_args()
    args = vars(args)

    # every subparser has an action arg specifying which action to perform
    action = args.pop('action')
    # `args.model` will be the key name of one of the models
    model_name = args.pop('model')
    PredictorClass = PREDICTORS[model_name]
    predictor = PredictorClass(target=args['target'], target_hours=np.arange(1, args['target_hours'] + 1))

    # do a bit of preprocessing with the tuning argument
    if 'no_tune' in args:
        args['tune'] = not args.pop('no_tune')
    else:
        args['tune'] = True

    if action == 'train':
        save_path = predictor.train(
            start=args['start'],
            stop=args['stop'],
            tune=args['tune'],
            num_folds=args['num_folds']
        )
        print(f'Model trained successfully.  Saved to {save_path}')

    elif action == 'evaluate':
        scores = predictor.cross_validate(
            start=args['start'],
            stop=args['stop'],
            num_folds=args['num_folds'],
            metrics=DEFAULT_METRICS
        )
        # report the mean scores for each metric
        for key in scores:
            print("Mean %s: %0.4f" % (key, scores[key]))

    elif action == 'predict':
        reftime = args['reftime']
        if 'latest' in args:
            reftime = pd.Timestamp('now').floor('6h')
        predictions = predictor.predict(
            reftime=reftime,
        )
        summary_path, prediction_path = predictor.write_prediction(
            predictions,
            summary_dir=args['summary_dir'],
            output_dir=args['output_dir']
        )
        print(f'Summary file written to {summary_path}\nPredictions written to {prediction_path}')

    else:
        print(f'ERROR: Action {action} is not defined for model {model_name}')
        parser.print_help()


if __name__ == '__main__':
    main()
