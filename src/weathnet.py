import torch
import torch.nn as nn
import torch.nn.functional as F

class LiLaBlock(nn.Module):
    def __init__(self, C_in, C_out):
        # input size should be : N*C_in*32*400
        # out size should be : N*C_out*32*400
        super(LiLaBlock, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_out, (7, 3), padding=(3, 1))
        self.conv2 = nn.Conv2d(C_in, C_out, (3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(C_in, C_out, (3, 3), padding=(2, 2), dilation=(2, 2))
        self.conv4 = nn.Conv2d(C_in, C_out, (3, 7), padding=(1, 3))
        self.conv5 = nn.Conv2d(4*C_out, C_out, (1, 1))
    
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x4 = F.relu(self.conv4(x))
        x_out = torch.cat((x1, x2, x3, x4), 1)
        return F.relu(self.conv5(x_out))


class WeatherNet(nn.Module):
    def __init__(self, C_in=2, C_out=3):
        super().__init__()
        self.lilablock1 = LiLaBlock(C_in, 32)
        self.lilablock2 = LiLaBlock(32, 64)
        self.lilablock3 = LiLaBlock(64, 96)
        self.lilablock4 = LiLaBlock(96, 96)
        self.drop = nn.Dropout(p=0.5)
        self.lilablock5 = LiLaBlock(96, 64)
        self.conv = nn.Conv2d(64, C_out, (1, 1))
    
    def forward(self, img):
        x = self.lilablock1.forward(img)
        x = self.lilablock2.forward(x)
        x = self.lilablock3.forward(x)
        x = self.lilablock4.forward(x)
        x = self.drop(x)
        x = self.lilablock5.forward(x)
        return F.relu(self.conv(x))
