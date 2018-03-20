from torch import nn


class MLP(nn.Module):
    def __init__(self, *chans):
        super().__init__()
        layers = []
        n = len(chans)
        for i in range(n-1):
            full = nn.Linear(chans[i], chans[i+1])
            relu = nn.ReLU(inplace=True)
            layers += [full, relu]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform(m.weight)
                nn.init.constant(m.bias, 0)
        return self
