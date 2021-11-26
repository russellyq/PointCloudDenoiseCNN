import torch
import torch.nn as nn
import torch.nn.functional as F

# input size should be : 1*32*400

class _LiLaBlock(nn.Module):
    def __init__(self):
        super(_LiLaBlock, self).__init__()
        