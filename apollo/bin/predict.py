import argparse
import pandas as pd
import numbers

import apollo.datasets.nam as nam
import apollo.models  # makes all model classes discoverable
from apollo.models.base import list_trained_models
from apollo.models.base import load as load_model
from apollo.output import write_csv, write_json


def main():
    model_names = list_trained_models()
    parser = argparse.ArgumentParser(
        description='Apollo Model Prediction Tool',
        argument_default=argparse.SUPPRESS,
    )
    # specify the model used to make predictions
    parser.add_argument('name', type=str, choices=model_names,
                        help='The name of the saved model used to generate '
                             'the prediction.')

    parser.add_argument('--reftime', '-r', default='2018-01-01 00:00', type=str,
                        help='The reftime for which predictions should be made.'
                             ' Any string accepted by pandas Timestamp'
                             ' constructor will work.'
                             ' Ignored if the `latest` flag is set.')

    parser.add_argument('--latest', '-l', action='store_true',
                        help='If set, a prediction will be generated for the '
                             'past reftime which is closest to the '
                             'current datetime.')

    parser.add_argument('--out_path', '-o', default=None, type=str,
                        help='The directory where predictions will be written.')

    parser.add_argument('--csv', '-c', action='store_true',
                        help='If set, predictions will be written as a CSV file'
                             ' instead of JSON.')

    # parse args
    args = parser.parse_args()
    args = vars(args)

    reftime = pd.Timestamp(args['reftime'])
    if 'latest' in args:
        reftime = pd.Timestamp('now').floor('6h')
    formatted_reftime = reftime.strftime('%Y_%m_%d-%H:%M')

    # ensure data for the requested reftime is cached
    print(f'Caching NAM data for reftime {reftime}...')
    nam.open(reftime - pd.Timedelta(6, 'h'), reftime)

    print('Generating predictions...')
    model_name = args['name']
    model = load_model(model_name)
    forecast = model.forecast(reftime)

    forecast_hours = model.data_kwargs["target_hours"]
    if isinstance(forecast_hours, numbers.Number):
        first_hour, last_hour = forecast_hours, forecast_hours
    else:
        first_hour, last_hour = forecast_hours[0], forecast_hours[-1]

    description = f'Predicted irradiance for {model.data_kwargs["target"]} ' \
                  f'for future hours {first_hour} through {last_hour}. ' \
                  f'Prediction generated by a {model.__class__.__name__} model.'

    print('Writing predictions to disk...')
    out_path = args['out_path']
    if 'csv' in args:
        output_file = write_csv(forecast=forecast,
                                name=f'{model_name}-{formatted_reftime}',
                                out_path=out_path)
    else:
        output_file = write_json(forecast=forecast,
                                 reftime=reftime,
                                 source=model_name,
                                 name=f'{model_name}-{formatted_reftime}',
                                 description=description,
                                 out_path=out_path)

    print(f'Wrote {output_file}')


if __name__ == '__main__':
    main()
