import torch
from torch.nn import Conv2d

class maskedConv2d(Conv2d):
    ''' MASKED CONVOLUTION '''
    
    def __init__(self, *args, **kwargs):
        
        # initialize the conv2d base class
        super().__init__(*args, **kwargs)
        
        # create a mask
        self.register_buffer('mask', self.weight.data.clone())
        
        # get the size of the weights
        _,d,h,w = self.weight.size()
        
        # make the default mask
        # ordering is applied in the forward method so we don't need
        # to worry about it here
        self.mask[:,:,h//2, w//2:] = 0
        self.mask[:,:,h//2 + 1:,:] = 0
        
    def forward(self, x, ordering):
        ''' 
        Ordering is an integer between 0 and 8. I'm not doing any error checking
        so please be respectful and don't pass in anything else.
        
        ordering == 0 is for inference, no mask is applied
        ordering == 1..8 correspond to particular raster scan orderings
        '''

        if ordering == 1:
            self.weight.data *= self.mask
        elif ordering == 2:
            self.weight.data *= torch.flip(self.mask, [3])
        elif ordering == 3:
            self.weight.data *= torch.rot90(self.mask, 1, [2,3])
        elif ordering == 4:
            self.weight.data *= torch.flip(torch.rot90(self.mask, 1, [2,3]), [2])
        elif ordering == 5:
            self.weight.data *= torch.rot90(self.mask, 2, [2,3])
        elif ordering == 6:
            self.weight.data *= torch.flip(torch.rot90(self.mask, 2, [2,3]), [3])
        elif ordering == 7:
            self.weight.data *= torch.rot90(self.mask, 3, [2,3])
        elif ordering == 8:
            self.weight.data *= torch.flip(torch.rot90(self.mask, 3, [2,3]), [3])
        
        
        return super().forward(x)
        