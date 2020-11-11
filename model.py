import torch
import torch.nn as nn
import torch.nn.functional as F
from maskedconv import maskedConv2d, shiftedMaskedConv2d
from attention_layer import attentionLayer


class stem(nn.Module):
    ''' The convolutional stem (h)'''
    def __init__(self, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(4, 64, kernel_size=3, padding=1, stride=stride) # potsdam RGBIR: 4 input channels
#         self.conv = nn.Conv2d(2, 64, kernel_size=3, padding=1) # radarsat: 2 input channels
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
        self.conv1 = shiftedMaskedConv2d(in_channels, 2*in_channels, kernel_size=3, padding=0)
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
    def __init__(self, in_channels, out_channels, upsample=2):
        super().__init__()
        self.upsample = upsample
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.soft = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=(self.upsample,self.upsample), \
                          mode='bilinear', align_corners=False)
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
#         self.resblock3 = AR_residual_block(256)
#         self.resblock4 = AR_residual_block(512)
#         self.decoder = decoder(1024, 3)
        
    def forward(self, x, ordering):
        x = self.stem(x)
        x = self.resblock1(x, ordering)
        x = self.resblock2(x, ordering)
#         x = self.resblock3(x, ordering)
#         x = self.resblock4(x, ordering)
        return self.decoder(x)

class ARSegmentationNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = stem(stride=2)
        self.attn = attentionLayer(64, 64, 64, 50, 50)
        self.resblock1 = AR_residual_block(64)
        self.resblock2 = AR_residual_block(128)
        self.decoder = decoder(256, 3, upsample=4)
        
    def forward(self, x, ordering):
        x = self.stem(x)
        x = self.attn(x, ordering)
        x = self.resblock1(x, ordering)
        x = self.resblock2(x, ordering)

        return self.decoder(x)

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == maskedConv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)