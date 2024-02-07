"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

from networks.res2net import res2net50
from networks.resnet import resnet50
from networks.cabm_resnet import ResidualNet
from networks.intersection import build_erase
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
class Dblock_more_dilate(nn.Module):
    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out

class BAM_LinkNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(BAM_LinkNet50, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = res2net50(pretrained=True)
        # resnet1 = ResidualNet('ImageNet', 50, 1000, att_type='BAM', path='/home/lxy/new_experiments/prelogs/RESNET50_IMAGENET_BAM_best.pth.tar')
        resnet1 = ResidualNet('ImageNet', 50, 1000, att_type='BAM',path=r'./prelogs/RESNET50_IMAGENET_BAM_best.pth.tar')
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.bam1 = resnet1.bam1
        self.bam2 = resnet1.bam2
        self.bam3 = resnet1.bam3
        

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e1=self.bam1(e1)
        e2 = self.encoder2(e1)
        e2=self.bam2(e2)
        e3 = self.encoder3(e2)
        e3=self.bam3(e3)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)


class GAMSNet_Noskip(nn.Module):
    def __init__(self, num_classes=1):
        super(GAMSNet_Noskip, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = res2net50(pretrained=True)
        # resnet1 = ResidualNet('ImageNet', 50, 1000, att_type='BAM', path='/home/lxy/new_experiments/prelogs/RESNET50_IMAGENET_BAM_best.pth.tar')
        resnet1 = ResidualNet('ImageNet', 50, 1000, att_type='BAM',
                              path=r'./prelogs/RESNET50_IMAGENET_BAM_best.pth.tar')
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.bam1 = resnet1.bam1
        self.bam2 = resnet1.bam2
        self.bam3 = resnet1.bam3

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e1 = self.bam1(e1)
        e2 = self.encoder2(e1)
        e2 = self.bam2(e2)
        e3 = self.encoder3(e2)
        e3 = self.bam3(e3)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)

class BAM_LinkNet50_T(nn.Module):
    def __init__(self, num_classes=1):
        super(BAM_LinkNet50_T, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = res2net50(pretrained=True)
        # resnet1 = ResidualNet('ImageNet', 50, 1000, att_type='BAM', path='/home/lxy/new_experiments/prelogs/RESNET50_IMAGENET_BAM_best.pth.tar')
        resnet1 = ResidualNet('ImageNet', 50, 1000, att_type='BAM',
                              path=r'./prelogs/RESNET50_IMAGENET_BAM_best.pth.tar')
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.bam1 = resnet1.bam1
        self.bam2 = resnet1.bam2
        self.bam3 = resnet1.bam3

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e1 = self.bam1(e1)
        e2 = self.encoder2(e1)
        e2 = self.bam2(e2)
        e3 = self.encoder3(e2)
        e3 = self.bam3(e3)
        e4 = self.encoder4(e3)


        return e1, e2,e3,e4


class GAMSNet_SOA(nn.Module):
    def __init__(self, num_classes=1):
        super(GAMSNet_SOA, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = res2net50(pretrained=True)
        # resnet1 = ResidualNet('ImageNet', 50, 1000, att_type='BAM', path='/home/lxy/new_experiments/prelogs/RESNET50_IMAGENET_BAM_best.pth.tar')
        resnet1 = ResidualNet('ImageNet', 50, 1000, att_type='BAM',
                              path=r'./prelogs/RESNET50_IMAGENET_BAM_best.pth.tar')
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.bam1 = resnet1.bam1
        self.bam2 = resnet1.bam2
        self.bam3 = resnet1.bam3

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        # self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3)
        # self.finalrelu2 = nonlinearity
        # self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)
        self.erase_channel = 16
        self.erase = build_erase(in_channel=256,erase_channel=16)
        self.deconv = nn.Sequential(nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )
        self.finalconv = nn.Conv2d(64 + self.erase_channel, 1, 1)
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e1 = self.bam1(e1)
        e2 = self.encoder2(e1)
        e2 = self.bam2(e2)
        e3 = self.encoder3(e2)
        e3 = self.bam3(e3)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2) # 256 , 256,256
        x1 = self.erase(d1)
        x2 = self.deconv(d1)
        x = torch.cat((x1, x2), dim=1)
        # x = self.deconv(x)
        x = self.finalconv(x)
        return torch.sigmoid(x)
        # out = self.finaldeconv1(d1)
        # out = self.finalrelu1(out)
        # out = self.finalconv2(out)
        # out = self.finalrelu2(out)
        # out = self.finalconv3(out)

        # return torch.sigmoid(d1)

class LinkNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet50, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)
class LinkNet50_T(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet50_T, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = resnet50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)


    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        return e1,e2,e3,e4
from networks.swin_transformer import SwinTransformer
class SwinT(nn.Module):
    def __init__(self, num_classes=1):
        super(SwinT, self).__init__()

        filters = [96, 192, 384, 768]
        self.backboon = SwinTransformer()
        self.backboon.init_weights('./weights/swin_tiny_patch4_window7_224_22k.pth')
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        # 32
        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 24, 3, stride=2)
        # self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(24, 24, 3)
        # self.finalrelu2 = nonlinearity
        # self.finalconv3 = nn.Conv2d(24, num_classes,2, padding=1)
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)
    def forward(self, x):
        # Encoder
        x4 = self.backboon(x)
        e4 = x4[3]
        e3 = x4[2]
        e2 = x4[1]
        e1 = x4[0]
        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        self.d0 = out
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        self.d4 = d4
        self.d3 = d3
        self.d2 = d2
        self.d1 = d1
        return torch.sigmoid(out)
class DinkNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(DinkNet50, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = res2net50(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
if __name__ == "__main__":
    import torch
    model = GAMSNet_SOA()
    input = torch.rand(2, 3, 512, 512)
    #
    y= model(input)
    print(y.size())
    # num_params = sum(p.numel() for p in model.parameters())
    # print("Total parameters: ", num_params)
    # torch.Size([2, 256, 128, 128])
    # torch.Size([2, 512, 64, 64])
    # torch.Size([2, 1024, 32, 32])
    # torch.Size([2, 2048, 16, 16])
#
#
# class LinkNet34(nn.Module):
#     def __init__(self, num_classes=1):
#         super(LinkNet34, self).__init__()
#
#         filters = [64, 128, 256, 512]
#         resnet = models.resnet34(pretrained=True)
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4
#
#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)
#
#     def forward(self, x):
#         # Encoder
#         x = self.firstconv(x)
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#         x = self.firstmaxpool(x)
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         e4 = self.encoder4(e3)
#
#         # Decoder
#         d4 = self.decoder4(e4) + e3
#         d3 = self.decoder3(d4) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#
#         return torch.sigmoid(out)
# class DinkNet34_less_pool(nn.Module):
#     def __init__(self, num_classes=1):
#         super(DinkNet34_more_dilate, self).__init__()
#
#         filters = [64, 128, 256, 512]
#         resnet = models.resnet34(pretrained=True)
#
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#
#         self.dblock = Dblock_more_dilate(256)
#
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
#
#     def forward(self, x):
#         # Encoder
#         x = self.firstconv(x)
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#         x = self.firstmaxpool(x)
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#
#         # Center
#         e3 = self.dblock(e3)
#
#         # Decoder
#         d3 = self.decoder3(e3) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
#
#         # Final Classification
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#
#         return F.sigmoid(out)
#
#
# class DinkNet34(nn.Module):
#     def __init__(self, num_classes=1, num_channels=3):
#         super(DinkNet34, self).__init__()
#
#         filters = [64, 128, 256, 512]
#         resnet = models.resnet34(pretrained=True)
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4
#
#         self.dblock = Dblock(512)
#
#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
#
#     def forward(self, x):
#         # Encoder
#         x = self.firstconv(x)
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#         x = self.firstmaxpool(x)
#         e1 = self.encoder1(x)  # torch.Size([4, 64, 192, 192])
#         e2 = self.encoder2(e1)  # torch.Size([4, 128, 96, 96])
#         e3 = self.encoder3(e2)  # torch.Size([4, 256, 48, 48])
#         e4 = self.encoder4(e3) # torch.Size([4, 512, 24, 24])
#
#         # Center
#         e4 = self.dblock(e4)
#
#         # Decoder
#         d4 = self.decoder4(e4) + e3
#         # batch, channel, height, width = e2.size()
#         # dd = F.interpolate(self.decoder3(d4), size=(height, width), mode='bilinear', align_corners=True)
#         d3 = self.decoder3(d4) + e2
#         # d3 = dd + e2
#         # batch, channel, heightee, widthee = e1.size()
#         # ddd = F.interpolate(self.decoder2(d3), size=(heightee, widthee), mode='bilinear', align_corners=True)
#         d2 = self.decoder2(d3) + e1
#         # d2 = ddd + e1
#         d1 = self.decoder1(d2)
#
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#
#         return torch.sigmoid(out)
#
#
# class DinkNet50(nn.Module):
#     def __init__(self, num_classes=1):
#         super(DinkNet50, self).__init__()
#
#         filters = [256, 512, 1024, 2048]
#         resnet = res2net50(pretrained=True)
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4
#
#         self.dblock = Dblock_more_dilate(2048)
#
#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
#
#     def forward(self, x):
#         # Encoder
#         x = self.firstconv(x)
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#         x = self.firstmaxpool(x)
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         e4 = self.encoder4(e3)
#
#         # Center
#         e4 = self.dblock(e4)
#
#         # Decoder
#         d4 = self.decoder4(e4) + e3
#         d3 = self.decoder3(d4) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#
#         return F.sigmoid(out)
#
#
# class DinkNet101(nn.Module):
#     def __init__(self, num_classes=1):
#         super(DinkNet101, self).__init__()
#
#         filters = [256, 512, 1024, 2048]
#         resnet = models.resnet101(pretrained=True)
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4
#
#         self.dblock = Dblock_more_dilate(2048)
#
#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
#
#     def forward(self, x):
#         # Encoder
#         x = self.firstconv(x)
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#         x = self.firstmaxpool(x)
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         e4 = self.encoder4(e3)
#
#         # Center
#         e4 = self.dblock(e4)
#
#         # Decoder
#         d4 = self.decoder4(e4) + e3
#         d3 = self.decoder3(d4) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#
#         return F.sigmoid(out)
#
# class Re_LinkNet50(nn.Module):
#     def __init__(self, num_classes=1):
#         super(Re_LinkNet50, self).__init__()
#
#         filters = [256, 512, 1024, 2048]
#         resnet = res2net50(pretrained=True)
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.firstmaxpool = resnet.maxpool
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4
#
#         self.decoder4 = DecoderBlock(filters[3], filters[2])
#         self.decoder3 = DecoderBlock(filters[2], filters[1])
#         self.decoder2 = DecoderBlock(filters[1], filters[0])
#         self.decoder1 = DecoderBlock(filters[0], filters[0])
#
#         self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
#         self.finalrelu1 = nonlinearity
#         self.finalconv2 = nn.Conv2d(32, 32, 3)
#         self.finalrelu2 = nonlinearity
#         self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)
#
#     def forward(self, x):
#         # Encoder
#         x = self.firstconv(x)
#         x = self.firstbn(x)
#         x = self.firstrelu(x)
#         x = self.firstmaxpool(x)
#         e1 = self.encoder1(x)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         e4 = self.encoder4(e3)
#
#         # Decoder
#         d4 = self.decoder4(e4) + e3
#         d3 = self.decoder3(d4) + e2
#         d2 = self.decoder2(d3) + e1
#         d1 = self.decoder1(d2)
#         out = self.finaldeconv1(d1)
#         out = self.finalrelu1(out)
#         out = self.finalconv2(out)
#         out = self.finalrelu2(out)
#         out = self.finalconv3(out)
#
#         return torch.sigmoid(out)


#
# class Dblock(nn.Module):
#     def __init__(self, channel):
#         super(Dblock, self).__init__()
#         self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
#         self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
#         self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
#         self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
#         # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def forward(self, x):
#         dilate1_out = nonlinearity(self.dilate1(x))
#         dilate2_out = nonlinearity(self.dilate2(dilate1_out))
#         dilate3_out = nonlinearity(self.dilate3(dilate2_out))
#         dilate4_out = nonlinearity(self.dilate4(dilate3_out))
#         # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
#         out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
#         return out