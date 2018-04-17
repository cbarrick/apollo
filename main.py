import argparse
from experiments import dtree_regressor


EXPERIMENTS = {
    'dtree': dtree_regressor.main
}


def info(args):
    """
    Print system info
    """
    import sys
    print('Python version:')
    print(sys.version)


def main():
    parser = argparse.ArgumentParser(description='Apollo: Solar Radiation Prediction')
    parser.add_argument('experiment', default='dtree', type=str, choices=list(EXPERIMENTS.keys()),
                        help='The name of the experiment to run \n[DEFAULT: dtree]')
    parser.add_argument('action', default='evaluate', type=str, choices=['train', 'evaluate', 'predict'],
                        help='The action to perform with this experiment.'
                             '\n\t - "train" : trains the model'
                             '\n\t - "evaluate" : uses cross-validation to output an approximation of the model\'s MAE'
                             '\n\t - "predict" : uses the model to make predictions about a new dataset'
                             '\n[DEFAULT: evaluate]')
    parser.add_argument('--cache_dir', '-c', default='/mnt/data6tb/chris/data', type=str,
                        help='The directory where the dataset is located.  If making new predictions, this directory'
                             'should contain a `NAM-NMM` subdirectory with downloaded NAM data.  '
                             'If training or evaluating a model, this directory should also contain a `GA-POWER` '
                             'subdirectory with the target data.  \n[DEFAULT: /media/data6tb/chris/data]')
    parser.add_argument('--save_dir', '-s', default='./models', type=str,
                        help='The directory where trained models will be serialized.  '
                             '\n[DEFAULT: ./models]')
    parser.add_argument('--prediction_dir', '-p', default='./predictions', type=str,
                        help='The directory where predictions will be written.  Ignored if action != predict.'
                             '\n[DEFAULT: ./predictions]')
    parser.add_argument('--start_date', '-b', default='2017-01-01 00:00', type=str,
                        help='The start date of the dataset that you want to use.  Any string accepted by numpy\'s '
                             'datetime64 constructor will work.  The data should already be downloaded to '
                             'the <cache_dir>.  '
                             '\n[DEFAULT: 2017-01-01 00:00]')
    parser.add_argument('--end_date', '-e', default='2017-12-31 18:00', type=str,
                        help='The end date of the dataset that you want to use.  Any string accepted by numpy\'s '
                             'datetime64 constructor will work.  The data should already be downloaded to '
                             'the <cache_dir>.  '
                             '\n[DEFAULT: 2017-12-31 18:00]')
    parser.add_argument('--target_hour', '-o', default=24, type=int,
                        help='The prediction hour to target between 1 and 36.'
                             '\n[DEFAULT: 24]')
    parser.add_argument('--target_var', '-t', default='UGA-C-POA-1-IRR', type=str,
                        help='The variable from GA_POWER to target.'
                             '\n[DEFAULT: 24]')

    # Each subcommand has a `experiment` which gives the name of the experiment to run
    # Call that experiment's main function and pass the rest of `args` as kwargs.
    args = parser.parse_args()
    if hasattr(args, 'experiment'):
        args = vars(args)
        experiment_name = args.pop('experiment')
        func = EXPERIMENTS[experiment_name]
        func(**args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
