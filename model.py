import torch
import torch.nn as nn
import torch.nn.functional as F


class SELayer(nn.Module):
    """Squeeze-and-Excitation layer for channel-wise attention."""
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class ResSEBlock(nn.Module):
    """Residual Block with SE attention."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.se = SELayer(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.shortcut(x)
        x = self.conv(x)
        x = self.se(x)
        return self.relu(x + res)


class AttentionGate(nn.Module):
    """Attention Gate to filter skip connections."""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv1d(F_g, F_int, kernel_size=1), nn.BatchNorm1d(F_int))
        self.W_l = nn.Sequential(nn.Conv1d(F_l, F_int, kernel_size=1), nn.BatchNorm1d(F_int))
        self.psi = nn.Sequential(nn.Conv1d(F_int, 1, kernel_size=1), nn.BatchNorm1d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, l):
        g1 = self.W_g(g)
        l1 = self.W_l(l)

        if g1.size(2) != l1.size(2):
            g1 = F.interpolate(g1, size=l1.size(2), mode='linear', align_corners=True)
        psi = self.relu(g1 + l1)
        psi = self.psi(psi)
        return l * psi


class SmartCutterUNet(nn.Module):
    def __init__(self, n_channels=160, n_classes=1):
        super().__init__()

        self.norm = nn.LayerNorm(n_channels)

        # Encoder
        self.inc = ResSEBlock(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool1d(2), ResSEBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool1d(2), ResSEBlock(128, 256))

        # Bottleneck with Dilation for context
        self.bot = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            ResSEBlock(512, 512)
        )

        # Decoder with Attention
        self.att2 = AttentionGate(F_g=512, F_l=128, F_int=64)
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv2 = ResSEBlock(512 + 128, 128)

        self.att1 = AttentionGate(F_g=128, F_l=64, F_int=32)
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv1 = ResSEBlock(128 + 64, 64)

        self.outc = nn.Conv1d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Layer norm
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)

        x1 = self.inc(x)        # 64
        x2 = self.down1(x1)     # 128
        x3 = self.down2(x2)     # 256
        x4 = self.bot(x3)       # 512

        # Up 1
        g2 = self.up2(x4)
        a2 = self.att2(g=x4, l=x2)
        if g2.size(2) != a2.size(2): g2 = F.interpolate(g2, size=a2.size(2))
        x = torch.cat([g2, a2], dim=1)
        x = self.conv2(x)

        # Up 2
        g1 = self.up1(x)
        a1 = self.att1(g=x, l=x1)
        if g1.size(2) != a1.size(2): g1 = F.interpolate(g1, size=a1.size(2))
        x = torch.cat([g1, a1], dim=1)
        x = self.conv1(x)

        return self.sigmoid(self.outc(x))