import logging

import numpy as np
import torch

import networks as N


logger = logging.getLogger(__name__)


class VggBlock2d(N.Module):
    def __init__(self, *chans):
        super().__init__()
        layers = []
        n = len(chans)
        for i in range(n-1):
            conv = N.Conv2d(chans[i], chans[i+1], kernel_size=3, stride=1, padding=1)
            relu = N.ReLU(inplace=True)
            layers += [conv, relu]
        layers += [N.MaxPool2d(kernel_size=2, stride=2)]
        self.layers = N.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _VggBase(N.Module):
    def __init__(self, cnn, shape, ndim):
        super().__init__()
        n = int(np.ceil(shape[1] / 2 ** len(cnn)))
        m = int(np.ceil(shape[2] / 2 ** len(cnn)))
        self.cnn = cnn
        self.frontend = N.MLP(512*n*m, 4096, 4096, ndim)
        self.reset()

    def reset(self):
        for m in self.modules():
            if isinstance(m, (N.Conv2d, N.Linear)):
                N.init.kaiming_uniform(m.weight)
                N.init.constant(m.bias, 0)
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
