from torch import nn


class LRN(nn.Module):
    '''A local response normalization layer.

    Written by @jiecaoyu on GitHub:
    https://github.com/pytorch/pytorch/issues/653
    '''
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, cross_channel=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.cross_channel = cross_channel

        if self.cross_channel:
            self.average = nn.AvgPool3d(
                kernel_size=(local_size, 1, 1),
                stride=1,
                padding=(int((local_size-1.0)/2), 0, 0),
            )
        else:
            self.average=nn.AvgPool2d(
                kernel_size=local_size,
                stride=1,
                padding=int((local_size-1.0)/2),
            )

    def forward(self, x):
        if self.cross_channel:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x
