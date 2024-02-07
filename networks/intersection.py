import torch
import torch.nn as nn
import torch.nn.functional as F

class _Asterisk_Erase(nn.Module):
    def __init__(self, in_channels, n_filters, cov_size):
        super(_Asterisk_Erase, self).__init__()
        self.cov_size = cov_size
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.weight1 = nn.Parameter(
            F.pad(torch.randint(0, 2, size=(in_channels // 8, in_channels // 4, 1, cov_size-2),
                                dtype=torch.float32),(1,1,0,0,0,0),value=1),requires_grad=False)
        self.weight2 = nn.Parameter(
            F.pad(torch.randint(0, 2, size=(in_channels // 8, in_channels // 4, cov_size-2, 1),
                                dtype=torch.float32), (0, 0, 1, 1, 0, 0), value=1), requires_grad=False)
        self.weight3 = nn.Parameter(
            F.pad(torch.randint(0, 2, size=(in_channels // 8, in_channels // 4, cov_size-2, 1),
                                dtype=torch.float32), (0, 0, 1, 1, 0, 0), value=1), requires_grad=False)
        self.weight4 = nn.Parameter(
            F.pad(torch.randint(0, 2, size=(in_channels // 8, in_channels // 4, 1, cov_size-2),
                                dtype=torch.float32), (1, 1, 0, 0, 0, 0), value=1), requires_grad=False)

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

        x1 = F.conv2d(x, self.weight1, padding=(0, self.cov_size//2))
        x2 = F.conv2d(x, self.weight2, padding=(self.cov_size//2, 0))
        x3 = self.inv_h_transform(F.conv2d(self.h_transform(x),self.weight3, padding=(self.cov_size//2, 0)))
        x4 = self.inv_v_transform(F.conv2d(self.v_transform(x), self.weight4, padding=(0, self.cov_size//2)))
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


class Asterisk_Erase(nn.Module):
    def __init__(self, in_ch, filters):
        super(Asterisk_Erase, self).__init__()
        asterisk_size = [5,9,13]
        in_channels=in_ch
        self.aster1 = _Asterisk_Erase(in_channels, filters//2, asterisk_size[0])
        self.aster2 = _Asterisk_Erase(in_channels, filters//2, asterisk_size[1])
        self.aster3 = _Asterisk_Erase(in_channels, filters//2, asterisk_size[2])
        self.conv1 = nn.Sequential(nn.Conv2d((filters//2)*3, filters, 1, bias=False),
                                     nn.BatchNorm2d(filters),
                                     nn.ReLU())
    def forward(self, x):
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

def build_erase(in_channel=64,erase_channel=16):
    return Asterisk_Erase(in_ch=in_channel, filters=erase_channel)
    #out:C38=8*7*6/2*3
if __name__ == "__main__":
    import torch
    decoder = build_erase(16)
    e4 = torch.rand(2, 64, 512, 512)
    x = decoder(e4)
    print(x.size())