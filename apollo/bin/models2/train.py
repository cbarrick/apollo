import argparse
import logging
import sys

import numpy as np
import pandas as pd

import apollo
from apollo import models2
from apollo.datasets import ga_power


logger = logging.getLogger(__name__)


def main(argv):
    parser = argparse.ArgumentParser(
        description='train a new model'
    )

    parser.add_argument(
        '--name',
        metavar='NAME',
        default=None,
        help='give a name to the model',
    )

    parser.add_argument(
        'template',
        metavar='TEMPLATE',
        help='the type of model (a template name or path to template file)',
    )

    parser.add_argument(
        'data_file',
        metavar='DATA',
        nargs='?',
        default=sys.stdin,
        help='the training data (a CSV, defaults to stdin)'
    )

    args = parser.parse_args(argv)

    logger.info('instantiating model from template')
    model = models2.from_template(args.template, name=args.name)

    logger.info('reading training data')
    targets = pd.read_csv(args.data_file, parse_dates=True, index_col=0)

    logger.info(f'training model with name {model.name}')
    model.fit(targets)
    path = model.save()
    print(path)
