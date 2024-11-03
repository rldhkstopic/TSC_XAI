import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(UNet, self).__init__()
        
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.middle = self.conv_block(512, 1024)

        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 각 채널의 평균을 구하여 1x1 형태로 축소
        self.output = nn.Linear(64, out_channels)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 인코딩
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))

        # 중간
        m = self.middle(F.max_pool2d(e4, 2))

        # 디코딩 + Skip Connection
        d4 = self.decoder4(torch.cat([F.interpolate(m, scale_factor=2), e4], dim=1))
        d3 = self.decoder3(torch.cat([F.interpolate(d4, scale_factor=2), e3], dim=1))
        d2 = self.decoder2(torch.cat([F.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.decoder1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))

        pooled = self.global_pool(d1).view(d1.size(0), -1)  # (batch_size, 64)
        class_logits = self.output(pooled)
        
        return class_logits


class U2Net(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(U2Net, self).__init__()

        # First U-Net
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.middle1 = self.conv_block(128, 256)

        # Second U-Net
        self.encoder3 = self.conv_block(256, 512)
        self.middle2 = self.conv_block(512, 1024)
        
        # Decoder of Second U-Net
        self.decoder3 = self.conv_block(1024 + 512, 512)
        self.decoder2 = self.conv_block(512 + 256, 256)
        
        # Decoder of First U-Net
        self.decoder1 = self.conv_block(256 + 128, 128)
        self.final_decoder = self.conv_block(128 + 64, 64)

        # Output layer
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # First U-Net Encoding
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        m1 = self.middle1(F.max_pool2d(e2, 2))

        # Second U-Net Encoding
        e3 = self.encoder3(F.max_pool2d(m1, 2))
        m2 = self.middle2(F.max_pool2d(e3, 2))

        # Second U-Net Decoding
        d3 = self.decoder3(torch.cat([F.interpolate(m2, scale_factor=2), e3], dim=1))
        d2 = self.decoder2(torch.cat([F.interpolate(d3, scale_factor=2), m1], dim=1))

        # First U-Net Decoding
        d1 = self.decoder1(torch.cat([F.interpolate(d2, scale_factor=2), e2], dim=1))
        final = self.final_decoder(torch.cat([F.interpolate(d1, scale_factor=2), e1], dim=1))

        return self.output(final)
