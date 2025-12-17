import torch
import torch.nn as nn


class UNet(nn.Module):
    """U-Net architecture for pixel-wise classification.

    Args:
        nn (nn.Module): The parent class.
    """

    def __init__(self, in_channels=3, classes=3, base_filters=64):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.enc4 = self._conv_block(base_filters * 4, base_filters * 8)

        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 8, base_filters * 16)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(
            base_filters * 16, base_filters * 8, 2, stride=2
        )
        self.dec4 = self._conv_block(base_filters * 16, base_filters * 8)

        self.upconv3 = nn.ConvTranspose2d(
            base_filters * 8, base_filters * 4, 2, stride=2
        )
        self.dec3 = self._conv_block(base_filters * 8, base_filters * 4)

        self.upconv2 = nn.ConvTranspose2d(
            base_filters * 4, base_filters * 2, 2, stride=2
        )
        self.dec2 = self._conv_block(base_filters * 4, base_filters * 2)

        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = self._conv_block(base_filters * 2, base_filters)

        # Final output
        self.out = nn.Conv2d(base_filters, classes, 1)

        self.pool = nn.MaxPool2d(2, 2)

    def _conv_block(self, in_channels, out_channels):
        """Double convolution block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return self.out(dec1)
