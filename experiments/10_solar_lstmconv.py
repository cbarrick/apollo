#!/usr/bin/env python3
import argparse
import logging
from types import SimpleNamespace

import numpy as np
import torch

import datasets as D
import estimators as E
import metrics as M
import networks as N
import optim as O


logger = logging.getLogger()


def seed(n):
    '''Seed the RNGs of stdlib, numpy, and torch.'''
    import random
    import numpy as np
    import torch
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n)


def main(**kwargs):
    kwargs.setdefault('epochs', 600)
    kwargs.setdefault('learning_rate', 0.001)
    kwargs.setdefault('patience', None)
    kwargs.setdefault('batch_size', 128)
    kwargs.setdefault('dry_run', False)
    kwargs.setdefault('name', None)
    kwargs.setdefault('seed', 1337)
    kwargs.setdefault('verbose', 'WARN')
    args = SimpleNamespace(**kwargs)

    logging.basicConfig(
        level=args.verbose,
        style='{',
        format='[{levelname:.4}][{asctime}][{name}:{lineno}] {msg}',
    )

    if args.name is None:
        now = np.datetime64('now')
        args.name = f'exp-{now}'
        logger.info(f'experiment name not given, defaulting to {args.name}')

    logger.debug('parameters of this experiment')
    for key, val in args.__dict__.items():
        logger.debug(f' {key:.15}: {val}')

    seed(args.seed)

    net = N.LstmConv4d(ndim=10)
    opt = O.Adam(net.parameters(), lr=args.learning_rate)
    loss = N.CrossEntropyLoss()
    model = E.Classifier(net, opt, loss, name=args.name, dry_run=args.dry_run)

    data = datasets[task]
    train, test = data.load()

    print(f'-------- Fitting --------')
    model.fit(train, epochs=args.epochs, patience=args.patience, batch_size=args.batch_size)
    print()

    print(f'-------- Scoring --------')
    scores = {
        'accuracy': M.Accuracy(),
        'true positives': M.TruePositives(),
        'false positives': M.FalsePositives(),
        'true negatives': M.TrueNegatives(),
        'false negatives': M.FalseNegatives(),
        'precision': M.Precision(),
        'recall': M.Recall(),
        'f-score': M.FScore(),
    }
    for metric, criteria in scores.items():
        score = model.test(test, criteria, batch_size=args.batch_size)
        print(f'{metric:15}: {score}')
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        add_help=False,
        argument_default=argparse.SUPPRESS,
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            'Runs an experiment.\n'
            '\n'
            'Note that the experiment is intended to be executed from the root of the\n'
            'repository using `python -m`:\n'
            '\n'
            '  python -m experiments.solar_lstmconv4d_v1\n'
            '\n'
        ),
    )

    group = parser.add_argument_group('Hyper-parameters')
    group.add_argument('-e', '--epochs', metavar='X', type=int)
    group.add_argument('-l', '--learning-rate', metavar='X', type=float)
    group.add_argument('-z', '--patience', metavar='X', type=int)

    group = parser.add_argument_group('Performance')
    group.add_argument('-b', '--batch-size', metavar='X', type=int)

    group = parser.add_argument_group('Debugging')
    group.add_argument('-d', '--dry-run', action='store_true')
    group.add_argument('-v', '--verbose', action='store_const', const='DEBUG')

    group = parser.add_argument_group('Other')
    group.add_argument('--seed')
    group.add_argument('--name', type=str)
    group.add_argument('--help', action='help')

    args = parser.parse_args()
    main(**vars(args))
