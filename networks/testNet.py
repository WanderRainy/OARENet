import torch.nn as nn
from networks.backbone import build_ResNet
from networks.decoder import build_decoder,build_decoder2
from networks.intersection import build_erase
import torch
from networks.dinknet import BAM_LinkNet50,BAM_LinkNet50_T,LinkNet50_T
from networks.swin_transformer import SwinTransformer

class GAMSNet_OAM(nn.Module):
    def __init__(self):
        super(GAMSNet_OAM, self).__init__()
        self.erase_channel=2
        self.resnet = BAM_LinkNet50_T()
        self.decoder = build_decoder()
        self.erase = build_erase(erase_channel=self.erase_channel)
        self.deconv = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )
        self.finalconv = nn.Conv2d(64+self.erase_channel,1,1)
    def forward(self,x):
        e1, e2, e3, e4=self.resnet(x)
        x = self.decoder(e1, e2, e3, e4)
        x1 = self.erase(x)
        x2 = self.deconv(x)
        x2 = torch.cat((x1, x2),dim=1)
        x = self.finalconv(x2)
        return torch.sigmoid(x)

class SwinT_A(nn.Module):
    def __init__(self):
        super(SwinT_A, self).__init__()

        filters = [96, 192, 384, 768]
        self.erase_channel = 0 #int(filters[0]/4)
        self.backboon = SwinTransformer()
        self.backboon.init_weights('./weights/swin_tiny_patch4_window7_224_22k.pth')
        self.decoder = build_decoder(filters) # int(filters[0]/4)
        # self.erase = build_erase(in_channel=int(filters[0]/4),erase_channel=self.erase_channel)
        self.deconv = nn.Sequential(nn.ConvTranspose2d(int(filters[0]/4), int(filters[0]/4), 3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(int(filters[0]/4)),
                                    nn.ReLU(),
                                    nn.Conv2d(int(filters[0]/4), int(filters[0]/4), 3,padding=1, bias=False),
                                    nn.BatchNorm2d(int(filters[0]/4)),
                                    nn.ReLU(),
                                    nn.Conv2d(int(filters[0]/4), int(filters[0]/4), 3,padding=1, bias=False),
                                    nn.BatchNorm2d(int(filters[0]/4)),
                                    nn.ReLU()
                                    )
        # self.finalconv = nn.Conv2d(int(filters[0]/4)+self.erase_channel,1,3,padding=1)
        self.finalconv =nn.Sequential(nn.Conv2d(int(filters[0] / 4)+self.erase_channel,int(filters[0] / 4), 3, padding=1),
                                      nn.BatchNorm2d(int(filters[0] / 4)),
                                      nn.ReLU(),
                                      # nn.Conv2d(int(filters[0] / 4), int(filters[0] / 4), 3,
                                      #           padding=1),
                                      # nn.BatchNorm2d(int(filters[0] / 4)),
                                      # nn.ReLU(),
                                      nn.Conv2d(int(filters[0] / 4), 1, 1))
    def forward(self,x):
        x4 = self.backboon(x)
        e4 = x4[3]
        e3 = x4[2]
        e2 = x4[1]
        e1 = x4[0]
        x = self.decoder(e1, e2, e3, e4)
        # x1 = self.erase(x)
        # x2 = self.deconv(x)
        # x = torch.cat((x1, x2),dim=1)
        x = self.deconv(x)
        x = self.finalconv(x)
        return torch.sigmoid(x)
class SwinT_OAM(nn.Module):
    def __init__(self):
        super(SwinT_OAM, self).__init__()

        filters = [96, 192, 384, 768]
        # self.erase_channel = 6 #int(filters[0]/4)
        self.backboon = SwinTransformer()
        self.backboon.init_weights('./weights/swin_tiny_patch4_window7_224_22k.pth')
        self.decoder = build_decoder(filters) # int(filters[0]/4)
        self.erase = build_erase(in_channel=int(filters[0]/4),erase_channel=int(filters[0]/4))
        self.deconv = nn.Sequential(nn.ConvTranspose2d(int(filters[0]/4), int(filters[0]/4), 3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(int(filters[0]/4)),
                                    nn.ReLU(),
                                    nn.Conv2d(int(filters[0]/4), int(filters[0]/4),1, bias=False),
                                    nn.BatchNorm2d(int(filters[0]/4)),
                                    nn.ReLU()
                                    )
        self.finalconv = nn.Conv2d(int(filters[0]/4),1,1)
    def forward(self,x):
        x4 = self.backboon(x)
        e4 = x4[3]
        e3 = x4[2]
        e2 = x4[1]
        e1 = x4[0]
        x = self.decoder(e1, e2, e3, e4)
        x1 = self.erase(x)
        x2 = self.deconv(x)
        x = x1+x2
        # x = self.deconv(x)
        x = self.finalconv(x)
        return torch.sigmoid(x)
        # return x1,x2
class LinkNet50_A(nn.Module):
    def __init__(self):
        super(LinkNet50_A, self).__init__()
        self.erase_channel=16

        self.resnet = LinkNet50_T()
        self.decoder = build_decoder()
        self.erase = build_erase(erase_channel=self.erase_channel)
        self.deconv = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.Conv2d(64, 64, 1, bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )
        self.finalconv = nn.Conv2d(64+self.erase_channel,1,1)
    def forward(self,x):
        e1, e2, e3, e4=self.resnet(x)
        x = self.decoder(e1, e2, e3, e4)
        x1 = self.erase(x)
        x2 = self.deconv(x)
        x = torch.cat((x1, x2),dim=1)
        # x = self.deconv(x)
        x = self.finalconv(x)
        return torch.sigmoid(x)
if __name__ == "__main__":
    import torch
    model = TestNet3()
    # model = LinkNet50_A()
    input = torch.rand(2, 3, 768, 768)
    y = model(input)
    print(y.size())
    # from thop import profile
    # temp = torch.rand(1, 3, 512, 512)
    # flops, params = profile(model, (temp,))
    # print()
