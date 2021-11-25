import torch
import torch.nn as nn
import torch.nn.functional as F

class _LiLaBlock(nn.Module):
    def __init__(self):
        super(_LiLaBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 7, 3)
        self.conv2 = nn.Conv2d(1, 3, 3)
        self.conv3 = nn.Conv2d(1, 3, 3, dilation=2)
        self.conv4 = nn.Conv2d(1, 3, 7)
        