import torch
import torch.nn as nn
import torch.nn.functional as F

class Asterisk(nn.Module):
    def __init__(self, in_channels, n_filters, cov_size):
        super(Asterisk, self).__init__()
        if cov_size == 5:
            kernel = 3
            dilation = 2
        elif cov_size == 9:
            kernel = 5
            dilation = 2
        elif cov_size == 13:
            kernel = 5
            dilation = 3
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, kernel), padding=(0, cov_size//2), dilation= dilation
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (kernel, 1), padding=(cov_size//2, 0), dilation= dilation
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (kernel, 1), padding=(cov_size//2, 0), dilation= dilation
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, kernel), padding=(0, cov_size//2), dilation= dilation
        )
        self.bn2 = nn.BatchNorm2d(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.ConvTranspose2d(in_channels // 4 + in_channels // 4, in_channels // 4 + in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(in_channels // 4 + in_channels // 4)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(
            in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn4 = nn.BatchNorm2d(n_filters)
        self.relu4 = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        return x
    def h_transform(self, x):
            shape = x.size()
            x = torch.nn.functional.pad(x, (0, shape[-1]))
            x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
            x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
            return x

    def inv_h_transform(self, x):
            shape = x.size()
            x = x.reshape(shape[0], shape[1], -1).contiguous()
            x = torch.nn.functional.pad(x, (0, shape[-2]))
            x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
            x = x[..., 0: shape[-2]]
            return x

    def v_transform(self, x):
            x = x.permute(0, 1, 3, 2)
            shape = x.size()
            x = torch.nn.functional.pad(x, (0, shape[-1]))
            x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
            x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
            return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
            x = x.permute(0, 1, 3, 2)
            shape = x.size()
            x = x.reshape(shape[0], shape[1], -1)
            x = torch.nn.functional.pad(x, (0, shape[-2]))
            x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
            x = x[..., 0: shape[-2]]
            return x.permute(0, 1, 3, 2)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(DecoderBlock, self).__init__()
        asterisk_size = [5,9,13]
        self.aster1 = Asterisk(in_channels, filters//2, asterisk_size[0])
        self.aster2 = Asterisk(in_channels, filters//2, asterisk_size[1])
        self.aster3 = Asterisk(in_channels, filters//2, asterisk_size[2])
        self.conv1 = nn.Sequential(nn.Conv2d((filters//2)*3, filters, 1, bias=False),
                                     nn.BatchNorm2d(filters),
                                     nn.ReLU())
    def forward(self, x, inp = False):
        x1 = self.aster1(x)
        x2 = self.aster2(x)
        x3 = self.aster3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv1(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self,filters = [256, 512, 1024, 2048]):
        super(Decoder, self).__init__()
        # filters = [96, 192, 384, 768]
        in_inplanes = filters[3]

        self.decoder4 = DecoderBlock(in_inplanes, filters[0])
        self.decoder3 = DecoderBlock(filters[1], int(filters[0]/2))
        self.decoder2 = DecoderBlock(filters[0], int(filters[0]/4))
        self.decoder1 = DecoderBlock(int(filters[0]/2), int(filters[0]/4))

        self.conv_e3 = nn.Sequential(nn.Conv2d(filters[2], filters[0], 1, bias=False),
                                       nn.BatchNorm2d(filters[0]),
                                       nn.ReLU())

        self.conv_e2 = nn.Sequential(nn.Conv2d(filters[1], int(filters[0]/2), 1, bias=False),
                                     nn.BatchNorm2d(int(filters[0]/2)),
                                     nn.ReLU())

        self.conv_e1 = nn.Sequential(nn.Conv2d(filters[0], int(filters[0]/4), 1, bias=False),
                                     nn.BatchNorm2d(int(filters[0]/4)),
                                     nn.ReLU())
        # self.deconv = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU())

        self._init_weight()


    def forward(self, e1, e2, e3, e4):
        d4 = torch.cat((self.decoder4(e4), self.conv_e3(e3)), dim=1)
        d3 = torch.cat((self.decoder3(d4), self.conv_e2(e2)), dim=1)
        d2 = torch.cat((self.decoder2(d3), self.conv_e1(e1)), dim=1)
        d1 = self.decoder1(d2)
        # x = self.deconv(d1)

        # x = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)

        return d1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class Decoder2(nn.Module):
    def __init__(self,filters = [256, 512, 1024, 2048]):
        super(Decoder2, self).__init__()
        # filters = [96, 192, 384, 768]
        in_inplanes = filters[3]

        self.decoder4 = DecoderBlock(in_inplanes, filters[0])
        self.decoder3 = DecoderBlock(filters[0], int(filters[0]/2))
        self.decoder2 = DecoderBlock(int(filters[0]/2), int(filters[0]/4))
        self.decoder1 = DecoderBlock(int(filters[0]/4), int(filters[0]/4))

        self.conv_e3 = nn.Sequential(nn.Conv2d(filters[2], filters[0], 1, bias=False),
                                       nn.BatchNorm2d(filters[0]),
                                       nn.ReLU())

        self.conv_e2 = nn.Sequential(nn.Conv2d(filters[1], int(filters[0]/2), 1, bias=False),
                                     nn.BatchNorm2d(int(filters[0]/2)),
                                     nn.ReLU())

        self.conv_e1 = nn.Sequential(nn.Conv2d(filters[0], int(filters[0]/4), 1, bias=False),
                                     nn.BatchNorm2d(int(filters[0]/4)),
                                     nn.ReLU())
        # self.deconv = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
        #                             nn.BatchNorm2d(64),
        #                             nn.ReLU())

        self._init_weight()


    def forward(self, e1, e2, e3, e4):
        d4 = self.decoder4(e4)+self.conv_e3(e3)
        d3 = self.decoder3(d4)+self.conv_e2(e2)
        d2 = self.decoder2(d3)+self.conv_e1(e1)
        d1 = self.decoder1(d2)
        # x = self.deconv(d1)

        # x = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)

        return d1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
def build_decoder2(filters = [256, 512, 1024, 2048]):
    return Decoder2(filters = filters)
def build_decoder(filters = [256, 512, 1024, 2048]):
    return Decoder(filters = filters)
if __name__ == "__main__":
    import torch
    decoder = build_decoder2(filters = [96, 192, 384, 768])
    # e1 = torch.rand(2, 256, 128, 128)
    # e2 = torch.rand(2, 512, 64, 64)
    # e3 = torch.rand(2, 1024, 32,32)
    # e4 = torch.rand(2, 2048, 16,16)
    e1 = torch.rand(2, 96, 128, 128)
    e2 = torch.rand(2, 192, 64, 64)
    e3 = torch.rand(2, 384, 32,32)
    e4 = torch.rand(2, 768, 16,16) # torch.Size([2, 24, 256, 256])
    x = decoder(e1,e2,e3,e4)
    print(x.size())
    # torch.Size([2, 64, 512, 512])



# input = torch.rand(2, 5, 10, 10)

