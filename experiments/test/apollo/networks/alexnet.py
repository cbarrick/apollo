import numpy as np

from torch import nn


class AlexNet(nn.Module):
    '''An AlexNet-like model based on the CIFAR-10 variant of AlexNet in Caffe.

    See:
        The Caffe version of this network
            https://github.com/BVLC/caffe/blob/1.0/examples/cifar10/cifar10_full.prototxt
        The Janowczyk and Madabhushi version, without dropout
            https://github.com/choosehappy/public/blob/master/DL%20tutorial%20Code/common/BASE-alexnet_traing_32w_db.prototxt
        The Janowczyk and Madabhushi version, with dropout
            https://github.com/choosehappy/public/blob/master/DL%20tutorial%20Code/common/BASE-alexnet_traing_32w_dropout_db.prototxt
    '''

    def __init__(self, shape=(3, 32, 32), ndim=10):
        super().__init__()

        # The Caffe version of this network uses LRN layers,
        # but Janowczyk and Madabhushi do not.
        self.features = nn.Sequential(
            nn.Conv2d(shape[0], 32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
        )

        # The Caffe version of the network uses a single linear layer without
        # activation. Janowczyk and Madabhushi use two linear layers, where the
        # first outputs a 64-vector. In their dropout network, _both_ layers
        # are followed by dropout and ReLU activation (it seems odd to activate
        # the final layer...). In the non-dropout network, they remove the
        # dropout _and_ activation. This is clearly a bug, since two linear
        # layers reduce to a single linear layer.
        n = int(np.ceil(shape[1] / 2 / 2 / 2))
        m = int(np.ceil(shape[2] / 2 / 2 / 2))
        self.classifier = nn.Sequential(
            nn.Linear(64*n*m, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, ndim),
        )

        self.reset()

    def reset(self):
        # Apply Kaiming initialization to conv and linear layers
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform(m.weight)
                nn.init.constant(m.bias, 0)
        return self

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
