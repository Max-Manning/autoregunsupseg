import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class attentionLayer(nn.Module):
    '''single-head attention
    This is a questionable implementation, who knows if it will work properly.'''
    def __init__(self, in_channels, dk, dv, H, W):
        super().__init__()
        
        self.dk = dk
        self.dv = dv
        
        # create a mask. Make sure H and W are correct for your input size or you will be sad
        mask = np.ones((H*W, H*W))
        mask = np.tril(mask)        
        self.register_buffer('mask', torch.from_numpy(mask).float())
        
        # the kernel size should just be one right???
        self.qconv = nn.Conv2d(in_channels, dk, kernel_size=1)
        self.kconv = nn.Conv2d(in_channels, dk, kernel_size=1)
        self.vconv = nn.Conv2d(in_channels, dv, kernel_size=1)
        
        # this shouldn't be necessary
#         self.qconv = nn.Conv2d(in_channels, dk, kernel_size=kernel_size, padding=padding)
#         self.kconv = nn.Conv2d(in_channels, dk, kernel_size=kernel_size, padding=padding)
#         self.vconv = nn.Conv2d(in_channels, dv, kernel_size=kernel_size, padding=padding)
        
        self.output_conv = nn.Conv2d(dv + in_channels, dv, kernel_size=1, stride=1)
        
    def forward(self, x, ordering):
        
        # get the shape of the input tensor
        B,C,H,W = x.shape
        
        # compute Q, K, V and reshape to the appropriate dimensions
        Q = self.qconv(x).reshape((B, 1, self.dk, H*W))
        K = self.kconv(x).reshape((B, 1, self.dk, H*W))
        V = self.vconv(x).reshape((B, 1, self.dv, H*W))
        
        # get value activations from the queries and keys
        logits = Q.transpose(2,3) @ K
        weights = F.softmax(logits, dim=-1)
    
        # apply a mask to make it causal
        if ordering == 1:
            weights = weights*self.mask
        elif ordering == 2:
            weights = weights*torch.flip(self.mask, [1])
        elif ordering == 3:
            weights = weights*torch.rot90(self.mask, 1, [0,1])
        elif ordering == 4:
            weights = weights*torch.flip(torch.rot90(self.mask, 1, [0,1]), [0])
        elif ordering == 5:
            weights = weights*torch.rot90(self.mask, 2, [0,1])
        elif ordering == 6:
            weights = weights*torch.flip(torch.rot90(self.mask, 2, [0,1]), [1])
        elif ordering == 7:
            weights = weights*torch.rot90(self.mask, 3, [0,1])
        elif ordering == 8:
            weights = weights*torch.flip(torch.rot90(self.mask, 3, [0,1]), [1])
        
        # get the output of the attention block
        o = weights @ V.transpose(2,3)
        
        # concatenate it channel-wise with the input
        oc = torch.cat((o.reshape((B,self.dv,H,W)), x), dim=1)
        
        # apply a 1x1 conv and return the result
        return self.output_conv(oc)
        



