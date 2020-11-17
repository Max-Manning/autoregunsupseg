import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class attentionLayer(nn.Module):
    '''
    Single-head self-attention block, similar to what's used in the pixelSNAIL paper
    
    Doesn't seem to work very well, so maybe I implemented it wrong? Who knows.
    '''
    def __init__(self, in_channels, dk, dv, H, W):
        super().__init__()
        
        self.dk = dk
        self.dv = dv
        
        # create a mask. Make sure H and W are correct for your input size or you will be sad
        # also this implementation assumes square images (H==W) so make sure of that
        mask = create_attention_masks(H)     
        self.register_buffer('mask', torch.from_numpy(mask).float())
        
        # 1x1 conv layers for creating queries, keys, values
        self.qconv = nn.Conv2d(in_channels, dk, kernel_size=1)
        self.kconv = nn.Conv2d(in_channels, dk, kernel_size=1)
        self.vconv = nn.Conv2d(in_channels, dv, kernel_size=1)
        
        # conv layer for creating the output of the attention layer
        self.att_conv = nn.Conv2d(dv, in_channels, kernel_size=1, stride=1)
        
        # conv layer for merging input and attention output after concatenation
        self.out_conv = nn.Conv2d(2*in_channels, in_channels, kernel_size=1, stride=1)
        
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
        
        # apply the attention mask to enforce causality
        if (ordering > 0) and (ordering < 9):
            weights = weights*self.mask[ordering-1,:,:]
        
        # get the output of the attention block
        o = weights @ V.transpose(2,3)
        o = self.att_conv(o.reshape((B,self.dv,H,W)))
        
        # concatenate the attention output with the input
        x = torch.cat((o,x), dim=1)
        
        # merge input and attention output using 1x1 conv and return
        return self.out_conv(x)
        
def create_attention_masks(H):
    '''creates the attention mask for each ordering'''
    
    # allocate space
    mask = np.zeros((8, H*H, H*H))
    
    # create attention masks for each ordering
    for ordering in range(1,9):
        
        a = np.arange(H*H).reshape((H, H))
        
        if ordering == 2:
            a = np.flip(np.rot90(a, 1), 0)  # CCW 90, flip V
        elif ordering==3:
            a = np.flip(a, 1)               # flip H
        elif ordering==4:
            a = np.rot90(a, 3)              # CCW 270
        elif ordering==5:
            a = np.rot90(a, 1)              # CCW 90
        elif ordering==6:
            a = np.flip(a, 0)               # flip V
        elif ordering==7:
            a = np.rot90(a, 2)              # CCW 180
        elif ordering==8:
            a = np.flip(np.rot90(a, 3), 0)  # CCW 270, flip V
        
        px_ord = a.flatten()

        for j in range(H*H):
            iid = px_ord[:j+1]
            mask[ordering-1,j,iid] = 1
    
    return mask