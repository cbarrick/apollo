import argparse
import logging
import sys

import numpy as np
import pandas as pd

import apollo
from apollo import models
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
        help='a name for the model, defaults to a UUID',
    )

    parser.add_argument(
        '-t',
        dest='named_template',
        action='store_true',
        help='load a builtin template rather than a template file',
    )

    parser.add_argument(
        '-o',
        metavar='DEST',
        dest='dest',
        default=None,
        help='output path for the model, defaults to the Apollo database',
    )

    parser.add_argument(
        'template',
        metavar='TEMPLATE',
        help='a path to template file or the name of a builtin template',
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
    if args.named_template:
        model = models.from_template_name(args.template, name=args.name)
    else:
        model = models.from_template(args.template, name=args.name)

    logger.info('reading training data')
    targets = pd.read_csv(args.data_file, parse_dates=True, index_col=0)

    logger.info(f'training model with name {model.name}')
    model.fit(targets)
    path = model.save(args.dest)
    print(path)
