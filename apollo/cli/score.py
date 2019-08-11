def description():
    import textwrap
    return textwrap.dedent('''\
    Evaluate the performance of a model.

    If the --from-file/-f option is given, MODEL must be a path to a model file.
    Otherwise, MODEL must be the name of a model in the Apollo database.

    The target file is a CSV with the same number and type of columns used when
    training the model. The file must have column headers and the first column
    must be the timestamps of the observations.
    ''')


def parse_args(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description=description(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-f',
        '--from-file',
        action='store_true',
        help='load the model from a file',
    )

    parser.add_argument(
        '-p',
        '--save-predictions',
        metavar='FILE',
        help='save model output'
    )

    parser.add_argument(
        'model',
        metavar='MODEL',
        help='the model to execute.'
    )

    parser.add_argument(
        'file',
        metavar='TARGETS',
        help='target data (defaults to stdin)'
    )

    return parser.parse_args(argv)


def log(message):
    import logging
    logger = logging.getLogger(__name__)
    logger.info(message)


def warn(message):
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(message)


def load_model(args):
    import sys
    import apollo
    from apollo import models

    try:
        log('loading model')
        if args.from_file:
            return models.load_model_at(args.model)
        else:
            return models.load_model(args.model)
    except FileNotFoundError:
        print(
            f'No such model: {args.model}\n' +
            f'Hint: You can list known models with `apollo ls models`',
            file=sys.stderr,
        )
        sys.exit(1)


def read_targets(args):
    log('loading targets')
    import pandas as pd
    return pd.read_csv(args.file, parse_dates=True, index_col=0)


def predict(model, index):
    log('executing model')
    return model.predict(index)


def score(targets, predictions, latlon=None):
    import apollo
    from apollo import metrics

    log('scoring predictions')
    assert (targets.columns == predictions.columns).all()

    missing = len(targets.index) - len(predictions.index)
    if missing != 0:
        warn(f'missing {missing} predictions')
        targets = targets.reindex(predictions.index)

    log('computing day-night scores')
    scores = metrics.all(targets, predictions)

    if latlon is not None:
        log('computing day-only scores')
        (lat, lon) = latlon
        index = predictions.index
        is_daylight = apollo.is_daylight(index, lat, lon)
        predictions = predictions[is_daylight]
        targets = targets[is_daylight]
        daytime_scores = metrics.all(targets, predictions)
        daytime_scores.index = daytime_scores.index + '_day_only'
        scores.index = scores.index + '_day_night'
        scores = scores.append(daytime_scores)

    scores.index.name = 'metric'
    return scores


def main(argv):
    import sys
    args = parse_args(argv)
    model = load_model(args)
    targets = read_targets(args)
    predictions = predict(model, targets.index)
    latlon = getattr(model, 'center', None)
    scores = score(targets, predictions, latlon)
    scores.to_csv(sys.stdout)
