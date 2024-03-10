"""
This file contains two definitions of LeNet:
1、the first Net is an common definition,
2、the other LeNet is for the convenience of pruning after adjusting the number of channels,
the cfg parameter, can be automatically adjusted

"""

import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet(nn.Module):
    def __init__(self, cfg=None):
        super(LeNet, self).__init__()
        if cfg is None:
            cfg = [6, 16, 120, 84, 10]
        self.cfg = cfg

        self.feature = self.make_layers(cfg, False)
        self.classifier = self.make_linear(cfg)

    def make_linear(self, cfg):
        layers = []
        in_channels = cfg[1] * 5 * 5
        for v in cfg[2:]:
            layers += [nn.Linear(in_channels, v)]
            in_channels = v
        return nn.Sequential(*layers)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 1
        for v in cfg[:2]:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=0, bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return F.log_softmax(y, dim=1)


if __name__ == "__main__":
    net = LeNet()
    x = torch.rand((1, 1, 28, 28))
    print(net)
    print(net(x))

    net = LeNet(cfg=[3, 8, 60, 42, 10])
    x = torch.rand((1, 1, 28, 28))
    print(net)
    print(net(x))
