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


class RecurrentResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, recurrent_steps = 2):
        super(RecurrentResidualBlock, self).__init__()
        self.recurrent_steps = recurrent_steps

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.in1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()


        self.recurrent = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU()
        )

        self.prelu = nn.PReLU()
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.InstanceNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.prelu(self.in1(self.conv1(x)))

        for _ in range(self.recurrent_steps):
            out = self.recurrent(out)

        out += self.shortcut(x)

        return out

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, recurrent_steps = 2):
        super(UpsampleBlock, self).__init__()
        self.recurrent_steps = recurrent_steps
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.recurrent = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU()
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(out_channels)
        )
    def forward(self, x, skip_features):
        x = self.upsample(x)
        x = torch.cat([x, skip_features], dim=1)

        out = x

        for _ in range(self.recurrent_steps):
            out = self.recurrent(out)

        out += self.shortcut(x)

        return out



class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class AttentionR2UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super(AttentionR2UNet, self).__init__()

        # Encoding block
        self.enc1 = RecurrentResidualBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = RecurrentResidualBlock(base_channels, base_channels*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = RecurrentResidualBlock(base_channels*2, base_channels*4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = RecurrentResidualBlock(base_channels*4, base_channels*8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottle neck
        self.bottleneck = RecurrentResidualBlock(base_channels*8, base_channels*16)

        # Decoding block
        self.up4 = UpsampleBlock(base_channels*16, base_channels*8, base_channels*8)
        self.att4 = AttentionGate(base_channels*8, base_channels*8, base_channels*4)
        self.up3 = UpsampleBlock(base_channels*8, base_channels*4, base_channels*4)
        self.att3 = AttentionGate(base_channels*4, base_channels*4, base_channels*2)
        self.up2 = UpsampleBlock(base_channels*4, base_channels*2, base_channels*2)
        self.att2 = AttentionGate(base_channels*2, base_channels*2, base_channels)
        self.up1 = UpsampleBlock(base_channels*2, base_channels, base_channels)
        self.att1 = AttentionGate(base_channels, base_channels, base_channels//2)

        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.up4(bottleneck, self.att4(enc4, bottleneck))
        dec3 = self.up3(dec4, self.att3(enc3, dec4))
        dec2 = self.up2(dec3, self.att2(enc2, dec3))
        dec1 = self.up1(dec2, self.att1(enc1, dec2))

        out = self.final_conv(dec1)

        return out
