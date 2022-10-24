""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class OnlyUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(OnlyUnet, self).__init__()
        # self.down=nn.Conv2d(in_feat,mid_feat,3,stride=4,padding=1)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.unet_seg = UNet(self.n_channels, self.n_classes, self.bilinear)
    def forward(self, x):
        #x:b,d,c,h,w
        input = x[0, :, :, :, :]
        #input:d,c,h,w
        out = self.unet_seg(input)
        real_out = out.unsqueeze(0)

        return real_out

    def init_weights(self):
        #logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std = 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def get_seg_model(cfg, **kwargs):
    model = OnlyUnet(cfg, **kwargs)
    model.init_weights()

    return model