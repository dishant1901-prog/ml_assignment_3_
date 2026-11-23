import torch.nn as nn


def down(cin, cout):
    return [nn.Conv2d(cin, cout, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


def up(cin, cout):
    return [nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True)]


class ConvAE(nn.Module):

    def __init__(self, bottleneck):
        super().__init__()
        self.bottleneck = bottleneck
        self.enc_conv = nn.Sequential(*down(3, 32), *down(32, 64), *down(64, 128), nn.Flatten())
        self.enc_fc = nn.Linear(128 * 4 * 4, bottleneck)
        self.dec_fc = nn.Sequential(nn.Linear(bottleneck, 128 * 4 * 4), nn.ReLU(inplace=True))
        self.dec_conv = nn.Sequential(*up(128, 64), *up(64, 32),
                                      nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1))

    def encode(self, x):
        return self.enc_fc(self.enc_conv(x))

    def decode(self, z):
        return self.dec_conv(self.dec_fc(z).view(-1, 128, 4, 4))

    def forward(self, x):
        return self.decode(self.encode(x))

