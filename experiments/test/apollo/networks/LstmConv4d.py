import numpy as np

from torch import nn


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LstmConv4d(nn.Module):
    def __init__(self, ndim=2):
        super().__init__()

        # The convolution network is a 3D version of VGG11.
        # It expects the input shape: N x 1 x 100 x 116 x 116,
        # in other words, don't pass in the sequence dimension.
        self.conv = nn.Sequential(
            # input: N x 1 x 100 x 116 x 116

            ConvBlock3d(1, 8),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # shape: N x 8 x 50 x 58 x 58

            ConvBlock3d(8, 16),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # shape: N x 16 x 25 x 29 x 29

            ConvBlock3d(16, 32),
            ConvBlock3d(32, 32),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=1),
            # shape: N x 32 x 13 x 15 x 15

            ConvBlock3d(32, 64),
            ConvBlock3d(64, 64),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=1),
            # shape: N x 64 x 7 x x 8 x 8

            ConvBlock3d(64, 64),
            ConvBlock3d(64, 64),
            nn.MaxPool3d(kernel_size=2, stride=2, padding=(1,0,0)),
            # shape: N x 64 x 4 x 4 x 4
        )

        # The LSTM has 2 layers at 512 hidden states each.
        # It expects the sequence dimension first, and the features to be flattened
        # i.e. the input shape: S x N x F
        self.rnn = nn.LSTM(
            input_size=64 * 4 * 4 * 4,
            hidden_size=512,
            num_layers=2,
            dropout=True,
            batch_first=True,
        )

        # The frontend network is a 2 layer MLP.
        self.frontend = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, ndim),
        )

        self.reset()

    def reset(self):
        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x, s=None):
        # input shape: N x T x 1 x 100 x 116 x 116
        n = x.size(0)
        t = x.size(1)
        x = x.view(n * t, 1, 100, 116, 116)
        x = self.conv(x)
        x = x.view(n, t, -1)
        x, s = self.rnn(x, s)
        x = x[:, -1, :]
        x = self.frontend(x)
        return x, s
