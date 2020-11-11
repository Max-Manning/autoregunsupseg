import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def create_attention_masks(H):
    
    mask = np.zeros((8, H*H, H*H))
    
    # create attention masks for each ordering
    for ordering in range(1,9):
        
        a = np.arange(H*H).reshape((H, H))
        
        if ordering == 2:
            a = np.flip(a, 1)               # 2
        elif ordering==3:
            a = np.rot90(a)                 # 3
        elif ordering==4:
            a = np.flip(np.rot90(a), 0)     # 4
        elif ordering==5:
            a = np.rot90(a, 2)              # 5
        elif ordering==6:
            a = np.flip(np.rot90(a, 2), 1)  # 6
        elif ordering==7:
            a = np.rot90(a, 3)              # 7
        elif ordering==8:
            a = np.flip(np.rot90(a, 3), 1)  # 8
        
        px_ord = a.flatten()

        for j in range(H*H):
            iid = px_ord[:j+1]
            mask[ordering-1,j,iid] = 1
    
    return mask

class attentionLayer(nn.Module):
    '''
    single-head self-attention block, similar to what's used in the pixelSNAIL paper.
    This is a questionable implementation, who knows if it will work properly.
    '''
    def __init__(self, in_channels, dk, dv, H, W):
        super().__init__()
        
        self.dk = dk
        self.dv = dv
        
        # create a mask. Make sure H and W are correct for your input size or you will be sad
        # also,,, I am currently assuming H==W so make sure of that
        mask = create_attention_masks(H)     
        self.register_buffer('mask', torch.from_numpy(mask).float())
        
        # the kernel size should just be one for all of these right???
        self.qconv = nn.Conv2d(in_channels, dk, kernel_size=1)
        self.kconv = nn.Conv2d(in_channels, dk, kernel_size=1)
        self.vconv = nn.Conv2d(in_channels, dv, kernel_size=1)
        self.output_conv = nn.Conv2d(dv, dv, kernel_size=1, stride=1)
        
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
        
        # apply the mask to enforce causality
        if (ordering > 0) and (ordering < 9):
            weights = weights*self.mask[ordering-1,:,:]
        
        # get the output of the attention block
        o = weights @ V.transpose(2,3)
        o = self.output_conv(o.reshape((B,self.dv,H,W)))
        
        # return the output of the attention block concatenated
        # with the input
        return torch.cat((o,x), dim=1)
        
def create_attention_masks(H):
    '''creates the attention for each ordering'''
    
    # allocate space
    mask = np.zeros((8, H*H, H*H))
    
    # create attention masks for each ordering
    for ordering in range(1,9):
        
        a = np.arange(H*H).reshape((H, H))
        
        if ordering == 2:
            a = np.flip(a, 1)               # 2
        elif ordering==3:
            a = np.rot90(a)                 # 3
        elif ordering==4:
            a = np.flip(np.rot90(a), 0)     # 4
        elif ordering==5:
            a = np.rot90(a, 2)              # 5
        elif ordering==6:
            a = np.flip(np.rot90(a, 2), 1)  # 6
        elif ordering==7:
            a = np.rot90(a, 3)              # 7
        elif ordering==8:
            a = np.flip(np.rot90(a, 3), 1)  # 8
        
        px_ord = a.flatten()

        for j in range(H*H):
            iid = px_ord[:j+1]
            mask[ordering-1,j,iid] = 1
    
    return mask