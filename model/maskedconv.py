import torch
from torch.nn import Conv2d
import torch.nn.functional as F

class shiftedMaskedConv2d(Conv2d):
    ''' 
    SHIFTED MASKED CONVOLUTION

    '''
    def __init__(self, *args, **kwargs):
        
        # initialize the conv2d base class
        super().__init__(*args, **kwargs)
        
        # Get the half kernel width for padding. We do the padding explicitly in the
        # forward() method since it depends on the selected ordering.
        self.pw = kwargs['kernel_size']//2
        
        # create a mask
        self.register_buffer('mask', self.weight.data.clone())
        
        # get the size of the weights
        b,c,h,w = self.weight.size()
        
        # make the default mask
        # ordering is applied in the forward method so we don't need
        # to worry about it here
        self.mask[:,:,:,:]=1 # set all the mask values to 1
        # self.mask[:,:,h//2 + self.pw, w//2:] = 0 # no access to center (current) pixel
        self.mask[:,:,h//2 + self.pw, w//2+1:] = 0 # allow access to center (current) pixel
        
    def forward(self, x, ordering):
        ''' 
        orderings == 1,2, .. 8 correspond to particular raster scan orderings to 
        be used during training.
        
        if ordering == 0 (or anything else) no mask is applied. Use this for inference.
        '''
        # depending on the ordering, apply the appropriate padding to the input
        # and get the transformed convolution mask
        
        if ordering == 1:
            x = F.pad(x, (self.pw,self.pw,2*self.pw,0)) # pad top
            mask = self.mask # no change
        elif ordering == 2:
            x = F.pad(x, (2*self.pw,0,self.pw,self.pw)) # pad left
            mask = torch.flip(torch.rot90(self.mask, 1, [2,3]), [2]) # CCW 90, flip V
        elif ordering == 3:
            x = F.pad(x, (self.pw,self.pw,2*self.pw,0)) # pad top
            mask = torch.flip(self.mask, [3]) # flip H
        elif ordering == 4:
            x = F.pad(x, (0,2*self.pw,self.pw,self.pw)) # pad right
            mask = torch.rot90(self.mask, 3, [2,3]) # CCW 270
        elif ordering == 5:
            x = F.pad(x, (2*self.pw,0,self.pw,self.pw)) # pad left
            mask = torch.rot90(self.mask, 1, [2,3]) # CCW 90
        elif ordering == 6:
            x = F.pad(x, (self.pw,self.pw,0,2*self.pw)) # pad bottom
            mask = torch.flip(self.mask, [2])# flip V
        elif ordering == 7:
            x = F.pad(x, (self.pw,self.pw,0,2*self.pw)) # pad bottom
            mask = torch.rot90(self.mask, 2, [2,3]) # CCW 180
        elif ordering == 8:
            x = F.pad(x, (0,2*self.pw,self.pw,self.pw)) # pad right
            mask = torch.flip(torch.rot90(self.mask, 3, [2,3]), [2]) # CCW 270, flip V
        else:
            x = F.pad(x, (self.pw,self.pw,self.pw,self.pw))
            mask = torch.ones_like(self.mask)
        
        # perform the convolution
        return F.conv2d(x, self.weight*mask, self.bias)

