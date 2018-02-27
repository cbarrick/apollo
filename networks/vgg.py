import logging

import numpy as np
import torch

import networks as N


logger = logging.getLogger(__name__)


class VggBlock2d(N.Module):
    def __init__(self, in_chans, *chans):
        super().__init__()
        layers = []

        # TODO: I should use input normalization
        # instead of batch norm on an input layer.
        bn = N.BatchNorm2d(in_chans)
        layers += [bn]

        for c in chans:
            conv = N.Conv2d(in_chans, c, kernel_size=3, stride=1, padding=1)
            bn = N.BatchNorm2d(c)
            relu = N.ReLU(inplace=True)
            layers += [conv, bn, relu]
            in_chans = c
        layers += [N.MaxPool2d(kernel_size=2, stride=2)]
        self.layers = N.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _VggBase(N.Module):
    def __init__(self, cnn, shape, ndim):
        super().__init__()
        (c, y, x) = shape

        # Adjust c, y, and x to their values after applying the CNN.
        # Each CNN block reduces the resolution by half.
        # The number of channels must be 512 at the output of the CNN.
        c = 512
        y = int(np.ceil(y / 2 ** len(cnn)))
        x = int(np.ceil(x / 2 ** len(cnn)))

        self.cnn = cnn
        self.frontend = N.MLP(512*y*x, 4096, 4096, ndim)
        self.reset()

    def reset(self, init_fn=N.init.kaiming_uniform):
        for m in self.modules():
            if not isinstance(m, (N.BatchNorm1d, N.BatchNorm2d, N.BatchNorm3d)):
                if hasattr(m, 'weight'): init_fn(m.weight)
                if hasattr(m, 'bias'): m.bias.data.fill_(0)
        return self

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.frontend(x)
        return x


class Vgg11(_VggBase):
    def __init__(self, shape=(3, 224, 224), ndim=1000):
        cnn = N.Sequential(
            VggBlock2d(shape[0], 64),
            VggBlock2d(64, 128),
            VggBlock2d(128, 256, 256),
            VggBlock2d(256, 512, 512),
            VggBlock2d(512, 512, 512),
        )
        super().__init__(cnn, shape, ndim)


class Vgg13(_VggBase):
    def __init__(self, shape=(3, 224, 224), ndim=1000):
        cnn = N.Sequential(
            VggBlock2d(shape[0], 64, 64),
            VggBlock2d(64, 128, 128),
            VggBlock2d(128, 256, 256),
            VggBlock2d(256, 512, 512),
            VggBlock2d(512, 512, 512),
        )
        super().__init__(cnn, shape, ndim)


class Vgg16(_VggBase):
    def __init__(self, shape=(3, 224, 224), ndim=1000):
        cnn = N.Sequential(
            VggBlock2d(shape[0], 64, 64),
            VggBlock2d(64, 128, 128),
            VggBlock2d(128, 256, 256, 256),
            VggBlock2d(256, 512, 512, 512),
            VggBlock2d(512, 512, 512, 512),
        )
        super().__init__(cnn, shape, ndim)


class Vgg19(_VggBase):
    def __init__(self, shape=(3, 224, 224), ndim=1000):
        cnn = N.Sequential(
            VggBlock2d(shape[0], 64, 64),
            VggBlock2d(64, 128, 128),
            VggBlock2d(128, 256, 256, 256, 256),
            VggBlock2d(256, 512, 512, 512, 512),
            VggBlock2d(512, 512, 512, 512, 512),
        )
        super().__init__(cnn, shape, ndim)
