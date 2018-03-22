'''A module for opening the MNIST digit recognition dataset
and other datasets which use the same file format.

The functions for reading the MNIST file format are taken from torchvision.
See https://github.com/pytorch/vision/commit/5861f14.
Credit to:
- Martin Raison (@martinraison)
- Ricky (@rtqichen)
- Adam Paszke (@apaszke)
'''

import codecs
import logging
from pathlib import Path

import numpy as np
import torch

import datasets as D


logger = logging.getLogger(__name__)


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        data = np.frombuffer(data, dtype=np.uint8, offset=8)
        data = data.astype(np.int64)
        data = data.reshape(length)
        return data


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        data = np.frombuffer(data, dtype=np.uint8, offset=16)
        data = data.astype(np.float64) / 255
        data = data.reshape(length, num_rows, num_cols)
        return data


class _MNISTDataset:
    def __init__(self, urls, data_dir, transform=None):
        self.urls = urls
        self.data_dir = Path(data_dir)
        self.transform = transform
        if not self.files['images'].exists(): self.download('images')
        if not self.files['labels'].exists(): self.download('labels')
        self.images = read_image_file(self.files['images'])
        self.labels = read_label_file(self.files['labels'])

    def __getitem__(self, i):
        label = self.labels[i]
        image = self.images[i]
        image = D.utils.apply_transform(image, self.transform)
        image = np.expand_dims(image, 0)
        image = torch.Tensor(image)
        return image, label

    def __len__(self):
        return len(self.images)

    def download(self, kind):
        D.utils.download(self.data_dir, self.urls[kind])
        D.utils.gunzip(self.gz_files[kind])

    @property
    def gz_files(self):
        return {
            kind: self.data_dir / url.rpartition('/')[2]
            for kind, url in self.urls.items()
        }

    @property
    def files(self):
        return {
            kind: gz_file.with_suffix('')
            for kind, gz_file in self.gz_files.items()
        }


class MNIST:
    urls = {
        'train': {
            'images': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'labels': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        },
        'test': {
            'images': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'labels': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
        },
    }

    def __init__(self, data_dir='./data/mnist'):
        self.data_dir = Path(data_dir)

    def load(self, transform=None):
        train = _MNISTDataset(self.urls['train'], (self.data_dir / 'train'), transform)
        test  = _MNISTDataset(self.urls['test'],  (self.data_dir / 'test'),  transform)
        return train, test


class FashionMNIST:
    urls = {
        'train': {
            'images': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
            'labels': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        },
        'test': {
            'images': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
            'labels': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
        },
    }

    def __init__(self, data_dir='./data/fashion_mnist'):
        self.data_dir = Path(data_dir)

    def load(self, transform=None):
        train = _MNISTDataset(self.urls['train'], (self.data_dir / 'train'), transform)
        test  = _MNISTDataset(self.urls['test'],  (self.data_dir / 'test'),  transform)
        return train, test
