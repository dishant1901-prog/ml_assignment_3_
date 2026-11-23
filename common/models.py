import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, hidden=(336, 336), in_dim=3072, n_classes=10, bn=True):
        super().__init__()
        layers = [nn.Flatten()]
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if bn:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def conv_bn(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CNN(nn.Module):

    def __init__(self, widths=(64, 128, 256), n_classes=10):
        super().__init__()
        a, b, c = widths
        self.features = nn.Sequential(
            *conv_bn(3, a), *conv_bn(a, a), nn.MaxPool2d(2),
            *conv_bn(a, b), *conv_bn(b, b), nn.MaxPool2d(2),
            *conv_bn(b, c), *conv_bn(c, c),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(c, n_classes)

    def forward(self, x):
        return self.fc(self.features(x))

    def embed(self, x):
        return self.features(x)

