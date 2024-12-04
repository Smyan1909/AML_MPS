import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,1,1)
        self.in2 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.InstanceNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.prelu(self.in1(self.conv1(x)))
        out = self.prelu(self.in2(self.conv2(out)))
        out += self.shortcut(x)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.conv = nn.Conv2d(out_channels + skip_channels, out_channels, 3, 1, 1)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x, skip_features):
        x = self.upsample(x)
        x = torch.cat([x + skip_features], dim=1)

        out = self.conv(x)
        out = self.in1(out)
        out = self.prelu(out)

        out += x

        return out

