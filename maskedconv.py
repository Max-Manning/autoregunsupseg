import torch
from torch.nn import Conv2d
import torch.nn.functional as F

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
        orderings == 1,2, .. 8 correspond to particular raster scan orderings to 
        be used during training.
        
        if ordering == 0 (or anything else) no mask is applied. Use this for inference.
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

class shiftedMaskedConv2d(Conv2d):
    ''' 
    SHIFTED MASKED CONVOLUTION

    '''
    
    def __init__(self, *args, **kwargs):
        
        # initialize the conv2d base class
        super().__init__(*args, **kwargs) # make sure padding=0 !!!
        
        # Get the half kernel width for padding. We do the padding explicitly in the
        # forward() method since it depends on the selected ordering. For this reason,
        # the 'padding' value of the parent Conv2d class is set to 0.
        self.pw = kwargs['kernel_size']//2
        
        # create a mask
        self.register_buffer('mask', self.weight.data.clone())
        
        # get the size of the weights
        _,d,h,w = self.weight.size()
        
        # make the default mask
        # ordering is applied in the forward method so we don't need
        # to worry about it here
        self.mask[:,:,h//2 + 1, w//2:] = 0
        self.mask[:,:,h//2 + 2:,:] = 0
        
    def forward(self, x, ordering):
        ''' 
        orderings == 1,2, .. 8 correspond to particular raster scan orderings to 
        be used during training.
        
        if ordering == 0 (or anything else) no mask is applied. Use this for inference.
        '''
        # depending on the ordering, apply the appropriate padding to the input, 
        # transform the convolution mask, and apply it to the convolution weights
        if ordering == 1:
            x = F.pad(x, (self.pw,self.pw,self.pw+1,self.pw-1))
            print(x.shape)
        elif ordering == 2:
            x = F.pad(x, (self.pw+1,self.pw-1,self.pw,self.pw))
            self.weight.data *= torch.flip(self.mask, [3])
        elif ordering == 3:
            x = F.pad(x, (self.pw,self.pw,self.pw+1,self.pw-1))
            self.weight.data *= torch.rot90(self.mask, 1, [2,3])
        elif ordering == 4:
            x = F.pad(x, (self.pw-1,self.pw+1,self.pw,self.pw))
            self.weight.data *= torch.flip(torch.rot90(self.mask, 1, [2,3]), [2])
        elif ordering == 5:
            x = F.pad(x, (self.pw+1,self.pw-1,self.pw,self.pw))
            self.weight.data *= torch.rot90(self.mask, 2, [2,3])
        elif ordering == 6:
            x = F.pad(x, (self.pw,self.pw,self.pw-1,self.pw+1))
            self.weight.data *= torch.flip(torch.rot90(self.mask, 2, [2,3]), [3])
        elif ordering == 7:
            x = F.pad(x, (self.pw,self.pw,self.pw-1,self.pw+1))
            self.weight.data *= torch.rot90(self.mask, 3, [2,3])
        elif ordering == 8:
            x = F.pad(x, (self.pw-1,self.pw+1,self.pw,self.pw))
            self.weight.data *= torch.flip(torch.rot90(self.mask, 3, [2,3]), [3])
        else:
            x = F.pad(x, (self.pw,self.pw,self.pw,self.pw))
        
        # perform the convolution
        return super().forward(x)