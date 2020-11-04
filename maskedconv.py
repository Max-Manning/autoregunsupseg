import torch
from torch.nn import conv2d

class maskedConv2d(Conv2d):
    ''' MASKED CONVOLUTION '''
    
    def __init__(self, masktype, *args, **kwargs):
        
        # initialize the conv2d base class
        super().__init__(*args, **kwargs)
        
        # create a mask
        self.register_buffer('mask', self.weight.data.clone())
        
        _, d,h,w = self.weight.size()
        
        if masktype == 'A':
            self.mask[:,:,h//2, w//2:] = 0
            self.mask[:,:,h//2 + 1:,:] = 0
        elif masktype == 'B':
            self.mask[:,:,h//2, w//2+1:] = 0
            self.mask[:,:,h//2 + 1:,:] = 0
        
    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)
        