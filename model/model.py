import torch
import torch.nn as nn
from model.components import stem, AR_residual_block, decoder
from model.attention_layer import attentionLayer

def init_weights(m):
    '''apply xavier initialization on model weights'''
    if hasattr(m, 'weight'):
        if len(m.weight.shape) > 1:
            torch.nn.init.xavier_normal_(m.weight)

class ARSegmentationNet2(nn.Module):
    '''2 residual blocks, no attention'''
    def __init__(self, in_channels=3, num_classes=3, stride=1):
        super().__init__()
        self.stem = stem(in_channels=in_channels, stride=stride)
        self.resblock1 = AR_residual_block(64)
        self.resblock2 = AR_residual_block(128)
        self.decoder = decoder(256, num_classes, upsample=2*stride)
        
    def forward(self, x, ordering):
        x = self.stem(x)
        x = self.resblock1(x, ordering)
        x = self.resblock2(x, ordering)
        return self.decoder(x)
    
class ARSegmentationNet2A(nn.Module):
    '''2 residual blocks, no attention'''
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.stem = stem(in_channels=in_channels, stride=2)
        self.resblock1 = AR_residual_block(64)
        self.attn = attentionLayer(128, 64, 64, 50)
        self.resblock2 = AR_residual_block(128)
        self.decoder = decoder(256, num_classes, upsample=4)
        
    def forward(self, x, ordering):
        x = self.stem(x)
        x = self.resblock1(x, ordering)
        x = self.attn(x, ordering)
        x = self.resblock2(x, ordering)
        return self.decoder(x)
    
class ARSegmentationNet3(nn.Module):
    '''3 residual blocks, no attention'''
    def __init__(self, in_channels=3, num_classes=3, stride=1):
        super().__init__()
        self.stem = stem(in_channels=in_channels, stride=stride)
        self.resblock1 = AR_residual_block(64)
        self.resblock2 = AR_residual_block(128)
        self.resblock3 = AR_residual_block(256)
        self.decoder = decoder(512, num_classes, upsample=2*stride)
        
    def forward(self, x, ordering):
        x = self.stem(x)
        x = self.resblock1(x, ordering)
        x = self.resblock2(x, ordering)
        x = self.resblock3(x, ordering)
        return self.decoder(x)
    
class ARSegmentationNet3A(nn.Module):
    '''3 residual blocks, no attention'''
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.stem = stem(in_channels=in_channels, stride=2)
        self.resblock1 = AR_residual_block(64)
        self.resblock2 = AR_residual_block(128)
        self.attn = attentionLayer(256, 128, 128, 50)
        self.resblock3 = AR_residual_block(256)
        self.decoder = decoder(512, num_classes, upsample=4)
        
    def forward(self, x, ordering):
        x = self.stem(x)
        x = self.resblock1(x, ordering)
        x = self.resblock2(x, ordering)
        x = self.attn(x, ordering)
        x = self.resblock3(x, ordering)
        return self.decoder(x)
    
class ARSegmentationNet4(nn.Module):
    '''4 residual blocks, no attention'''
    def __init__(self, in_channels=3, num_classes=3, stride=1):
        super().__init__()
        self.stem = stem(in_channels=in_channels, stride=stride)
        self.resblock1 = AR_residual_block(64)
        self.resblock2 = AR_residual_block(128)
        self.resblock3 = AR_residual_block(256)
        self.resblock4 = AR_residual_block(512)
        self.decoder = decoder(1024, num_classes, upsample=2*stride)
        
    def forward(self, x, ordering):
        x = self.stem(x)
        x = self.resblock1(x, ordering)
        x = self.resblock2(x, ordering)
        x = self.resblock3(x, ordering)
        x = self.resblock4(x, ordering)
        return self.decoder(x)


class ARSegmentationNet4A(nn.Module):
    '''3 residual blocks, no attention'''
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.stem = stem(in_channels=in_channels, stride=2)
        self.resblock1 = AR_residual_block(64)
        self.resblock2 = AR_residual_block(128)
        self.attn = attentionLayer(256, 128, 128, 50)
        self.resblock3 = AR_residual_block(256)
        self.resblock4 = AR_residual_block(512)
        self.decoder = decoder(1024, num_classes, upsample=4)
        
    def forward(self, x, ordering):
        x = self.stem(x)
        x = self.resblock1(x, ordering)
        x = self.resblock2(x, ordering)
        x = self.attn(x, ordering)
        x = self.resblock3(x, ordering)
        x = self.resblock4(x, ordering)
        return self.decoder(x)