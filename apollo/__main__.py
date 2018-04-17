import argparse
from experiments import dtree_regressor


EXPERIMENTS = {
    'dtree': dtree_regressor
}


def main():
    parser = argparse.ArgumentParser(
        description='Apollo: Solar Radiation Prediction with Machine Learning',
        argument_default=argparse.SUPPRESS,
    )

    # arguments that are common across all sub-commands

    parser.add_argument('--model', '-m', default='dtree', type=str, choices=list(EXPERIMENTS.keys()),
                        help='The name of the model that you would like to run.')

    parser.add_argument('--begin_date', '-b', default='2017-01-01 00:00', type=str,
                        help='The start date of the dataset that you want to use.  Any string accepted by numpy\'s '
                        'datetime64 constructor will work.  The data should already be downloaded to the <cache_dir>.')

    parser.add_argument('--end_date', '-e', default='2017-12-31 18:00', type=str,
                        help='The end date of the dataset that you want to use.  Any string accepted by numpy\'s '
                        'datetime64 constructor will work.  The data should already be downloaded to the <cache_dir>.')

    parser.add_argument('--target_hour', '-i', default=24, type=int,
                        help='The prediction hour to target.  Should be an integer between 1 and 36.')

    parser.add_argument('--target_var', '-t', default='UGA-C-POA-1-IRR', type=str,
                        help='The variable from GA_POWER to target.')

    parser.add_argument('--cache_dir', '-c', default='/mnt/data6tb/chris/data', type=str,
                        help='The directory where the dataset is located.  This directory should contain a `NAM-NMM` '
                             'subdirectory with downloaded NAM data. If training or evaluating a model, '
                             'this directory should also contain a `GA-POWER` subdirectory with the target data.')

    subcommands = parser.add_subparsers()

    # train
    train = subcommands.add_parser('train', argument_default=argparse.SUPPRESS, description='Train a model.')
    train.set_defaults(action='train')

    train.add_argument('--save_dir', '-s', default='./models', type=str,
                       help='The directory where trained models will be serialized. This directory will be created if'
                            ' it does not exist.')
    train.add_argument('--tune', '-p', action='store_true',
                       help='If set, hyperparameter tuning will be performed using a cross-validated grid search before'
                            'training on the specified dataset')
    train.add_argument('--num_folds', '-n', default=3, type=int,
                       help='If `tune` is enabled, the number of folds to use during the cross-validated grid search. ' 
                            'Ignored if tuning is disabled.')

    # evaluate
    evaluate = subcommands.add_parser('evaluate', argument_default=argparse.SUPPRESS,
                                      description='Evaluate a model using n-fold cross-validation')
    evaluate.set_defaults(action='evaluate')

    evaluate.add_argument('--num_folds', '-n', default=3, type=int,
                          help='The number of folds to use when computing cross-validated accuracy.')

    # TODO: add option to evaluate using several metrics

    # predict
    predict = subcommands.add_parser('predict', argument_default=argparse.SUPPRESS,
                                     description='Make predictions using a trained model.')
    predict.set_defaults(action='predict')

    predict.add_argument('--save_dir', '-s', default='./models', type=str,
                         help='The directory where trained models will be serialized. This directory will be created if'
                              ' it does not exist.')
    predict.add_argument('--output_dir', '-o', default='./predictions', type=str,
                         help='The directory where predictions will be written.')

    # parse args and invoke the correct experiment
    args = parser.parse_args()
    args = vars(args)

    # every subparser has an action arg specifying which action to perform
    action = args.pop('action')
    # argparse guarantees that `args.model` will be the key name of one of the experiments
    experiment = EXPERIMENTS[args.pop('model')]

    if action == 'train':
        save_path = experiment.train(**args)
        print(f'Model trained successfully.  Saved to {save_path}')
    elif action == 'evaluate':
        score = experiment.evaluate(**args)
        print('Average MAE: %0.4f' % score)
    elif action == 'predict':
        prediction_file = experiment.predict(**args)
        print(f'Output written to {prediction_file}')
    else:
        print(f'ERROR: Action {action} is not defined for model {experiment}')
        parser.print_help()


if __name__ == '__main__':
    main()
