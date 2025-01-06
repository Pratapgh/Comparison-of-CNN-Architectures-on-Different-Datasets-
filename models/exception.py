
import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Xception(nn.Module):
    def __init__(self, num_classes=10):
        super(Xception, self).__init__()
        self.entry = SeparableConv2d(3, 64)
        self.middle = SeparableConv2d(64, 128)
        self.exit = SeparableConv2d(128, 256)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
