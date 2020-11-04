import torch
import torch.nn as nn
import torch.nn.functional as F
from maskedconv import maskedConv2d

class stem(nn.Module):
    ''' The convolutional stem (h)'''
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return self.pool(x)

class AR_residual_block(nn.Module):
    '''Residual block for the autoregressive encoder section (gar)'''
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        # first sub-block
        self.conv1 = maskedConv2d(in_channels, 2*in_channels, kernel_size=3, padding=1)
        # masked conv is equivalent to regular conv for 1x1 kernel so I'll just use that
        self.conv2 = nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=1)
        
        # second sub-block
        self.conv3 = nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=1)
        
        # third sub-block
        self.conv5 = nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=1)
        self.conv6 = nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=1)
        
    def forward(self, x, ordering):
        # first sub-block
        residual = F.pad(x, (0,0,0, 0, 0, self.in_channels))
        x = F.relu(self.conv1(x, ordering))
        x = F.relu(self.conv2(x))
        x = x + residual
        
        # second sub-block
        residual = x
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x + residual 
        
        # third sub-block
        residual = x
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = x + residual
        
        return x
        
class decoder(nn.Module):
    '''The decoder section (d)'''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=(2,2), mode='bilinear')
        return self.soft(x)
    
class ARSegmentationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = stem()
        # am I getting this right? If each residual block doubles the number of
        # channels, we rapidly get to an appalling number
        self.resblock1 = AR_residual_block(64)
        self.resblock2 = AR_residual_block(128)
        self.decoder = decoder(256, 6)
        
    def forward(self, x, ordering):
        x = self.stem(x)
        x = self.resblock1(x, ordering)
        x = self.resblock2(x, ordering)
        return self.decoder(x)