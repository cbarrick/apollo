import argparse
import numpy as np
from apollo.prediction.SKPredictor import SKPredictor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.regression import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

MODELS = {
    'linreg': LinearRegression(),
    'svr': SVR(),
    'knn': KNeighborsRegressor(),
    'dtree': DecisionTreeRegressor(),
    'rf': RandomForestRegressor(),
    'gbt': XGBRegressor()
}

PARAM_GRIDS = {
    'linreg': None,
    'svr': {
        'estimator__C': np.arange(1.0, 1.6, 0.2),                  # penalty parameter C of the error term
        'estimator__epsilon': np.arange(0.4, 0.8, 0.1),            # width of the no-penalty region
        'estimator__kernel': ['rbf', 'sigmoid'],                   # kernel function
        'estimator__gamma': [1/1000]                               # kernel coefficient
    },
    'knn': {
        'estimator__n_neighbors': np.arange(3, 15, 2),             # k
        'estimator__weights': ['uniform', 'distance'],             # how are neighboring values weighted
    },
    'dtree': {
        'estimator__splitter': ['best', 'random'],                 # splitting criterion
        'estimator__max_depth': [None, 10, 20, 30],                # Maximum depth of the tree. None means unbounded.
        'estimator__min_impurity_decrease': np.arange(0.15, 0.40, 0.05)
    },
    'rf': {
        'estimator__n_estimators': [10, 50, 100, 250],
        'estimator__max_depth': [None, 10, 20, 30],                # Maximum depth of the tree. None means unbounded.
        'estimator__min_impurity_decrease': np.arange(0.15, 0.40, 0.05)
    },
    'gbt': {
        'estimator__learning_rate': np.arange(0.03, 0.07, 0.02),   # learning rate
        'estimator__n_estimators': [50, 100, 200, 250],            # number of boosting stages
        'estimator__max_depth': [3, 5, 10, 20],                    # Maximum depth of the tree. None means unbounded.
    }
}

DEFAULT_METRICS = {
    'mse': mean_squared_error,
    'mae': mean_absolute_error,
    'r2': r2_score
}


def main():
    parser = argparse.ArgumentParser(
        description='Apollo: Solar Radiation Prediction with Machine Learning',
        argument_default=argparse.SUPPRESS,
    )

    # arguments that are common across all sub-commands

    parser.add_argument('--model', '-m', default='dtree', type=str, choices=list(MODELS.keys()),
                        help='The name of the model that you would like to run.')

    parser.add_argument('--start', '-b', default='2017-01-01 00:00', type=str,
                        help='The start date of the dataset that you want to use. '
                             'Any string accepted by numpy\'s datetime64 constructor will work.')

    parser.add_argument('--stop', '-e', default='2017-12-31 18:00', type=str,
                        help='The end date of the dataset that you want to use. '
                             'Any string accepted by numpy\'s datetime64 constructor will work.')

    parser.add_argument('--target_hours', '-f', default=24, type=int,
                        help='Generate predictions for each our up to this hour. '
                             'Should be an integer between 1 and 36.')

    parser.add_argument('--target', '-t', default='UGA-C-POA-1-IRR', type=str,
                        help='The variable from GA_POWER to target.')

    parser.add_argument('--save_dir', '-s', default='./models', type=str,
                        help='The directory where trained models will be serialized. This directory will be created if'
                             ' it does not exist.')

    subcommands = parser.add_subparsers()

    # train
    train = subcommands.add_parser('train', argument_default=argparse.SUPPRESS, description='Train a model.')
    train.set_defaults(action='train')
    train.add_argument('--no_tune', '-p', action='store_true',
                       help='If set, hyperparameter tuning will NOT be performed during training.')
    train.add_argument('--num_folds', '-n', default=3, type=int,
                       help='If `tune` is enabled, the number of folds to use during the cross-validated grid search. '
                            'Ignored if tuning is disabled.')

    # evaluate
    evaluate = subcommands.add_parser('evaluate', argument_default=argparse.SUPPRESS,
                                      description='Evaluate a model using n-fold cross-validation')
    evaluate.set_defaults(action='evaluate')
    evaluate.add_argument('--num_folds', '-n', default=3, type=int,
                          help='The number of folds to use when computing cross-validated accuracy.')

    # predict
    predict = subcommands.add_parser('predict', argument_default=argparse.SUPPRESS,
                                     description='Make predictions using a trained model.')
    predict.set_defaults(action='predict')
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
    predictor = SKPredictor(name=model_name, estimator=MODELS[model_name], parameter_grid=PARAM_GRIDS[model_name],
                            target=args['target'], target_hours=np.arange(1, args['target_hours'] + 1))

    # do a bit of preprocessing with the tuning argument
    if 'no_tune' in args:
        args['tune'] = not args.pop('no_tune')
    else:
        args['tune'] = True

    if action == 'train':
        save_path = predictor.train(
            start=args['start'],
            stop=args['stop'],
            save_dir=args['save_dir'],
            tune=args['tune'],
            num_folds=args['num_folds']
        )
        print(f'Model trained successfully.  Saved to {save_path}')

    elif action == 'evaluate':
        scores = predictor.cross_validate(
            start=args['start'],
            stop=args['stop'],
            save_dir=args['save_dir'],
            num_folds=args['num_folds'],
            metrics=DEFAULT_METRICS
        )
        # report the mean scores for each metric
        for key in scores:
            print("Mean %s: %0.4f" % (key, scores[key]))

    elif action == 'predict':
        predictions = predictor.predict(
            start=args['start'],
            stop=args['stop'],
            save_dir=args['save_dir']
        )
        print(predictions.shape)
        summary_path, prediction_path = predictor.write_predictions(
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
