import torch
import torch.nn.functional as F

EPS = 1e-10

def MI_loss(out_o1, out_o2, T=0): 
    '''
    MI loss. Adapted from IID_segmentation_loss:
    https://github.com/xu-ji/IIC/blob/master/code/utils/segmentation/IID_losses.py
    
    Args:
        out_o1: output of the network using the first ordering
        out_o2: output of the network using the second ordering
        T: half window width for spatial invariance 
    '''
    
    out_o1 = out_o1.permute(1,0,2,3)
    out_o2 = out_o2.permute(1,0,2,3)
    
    # sum over everything except classes to get the joint probability distribution
    # over the classes
    p_i_j = F.conv2d(out_o1, weight=out_o2, padding=(T,T))
    p_i_j = p_i_j.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)
    
    # normalize
    current_norm = float(p_i_j.sum())
    p_i_j = p_i_j / current_norm
    
    # symmetrise
    p_i_j = (p_i_j + p_i_j.t()) / 2.
    
    # compute marginals
    p_i_mat = p_i_j.sum(dim=1).unsqueeze(1)  # k, 1
    p_j_mat = p_i_j.sum(dim=0).unsqueeze(0)  # 1, k
    
    # for log stability; tiny values cancelled out by mult with p_i_j anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_i_mat[(p_i_mat < EPS).data] = EPS
    p_j_mat[(p_j_mat < EPS).data] = EPS

    # maximise information
    loss = (-p_i_j * (torch.log(p_i_j) - torch.log(p_i_mat) - torch.log(p_j_mat))).sum()
    
    return loss


## note: ignore the edge losses, they're not related to the paper implementation

def edge_loss_4(output, image, beta=2):
    
    # edge penalty!!!
    
    # get the 4 directional gradients
    grad_st = torch.zeros((image.shape[0], image.shape[2], image.shape[3], 4))
    grad_st[:,:,:,0] = torch.sum(torch.abs(image - torch.roll(image,  1, 2)), dim=1)
    grad_st[:,:,:,1] = torch.sum(torch.abs(image - torch.roll(image, -1, 2)), dim=1)
    grad_st[:,:,:,2] = torch.sum(torch.abs(image - torch.roll(image,  1, 3)), dim=1)
    grad_st[:,:,:,3] = torch.sum(torch.abs(image - torch.roll(image, -1, 3)), dim=1)
    
    # find out where there are class boundaries
    labels = torch.argmax(output, 1)
    label_diffs = torch.zeros((image.shape[0], image.shape[2], image.shape[3], 4))
    label_diffs[:,:,:,0] = labels == torch.roll(labels,  1, 1)
    label_diffs[:,:,:,1] = labels == torch.roll(labels, -1, 1)
    label_diffs[:,:,:,2] = labels == torch.roll(labels,  1, 2)
    label_diffs[:,:,:,3] = labels == torch.roll(labels, -1, 2)
    
    # no edge penalty at the edges!!
    label_diffs[:, 0, :,:] = 0
    label_diffs[:,-1, :,:] = 0
    label_diffs[:, :, 0,:] = 0
    label_diffs[:, :,-1,:] = 0
    
    edge_pen = grad_st * label_diffs
    num_el = image.shape[0]*image.shape[1]*image.shape[2]*image.shape[3]
    return beta*torch.sum(torch.exp(-1 * torch.mul(edge_pen, edge_pen)))/num_el

def edge_loss_8(output, image, beta, K):
    
    # edge penalty!!!
    
    # get the 4 directional gradients
    grad_st = torch.zeros((image.shape[0], image.shape[2], image.shape[3], 8))
    grad_st[:,:,:,0] = torch.sum(torch.abs(image - torch.roll(image,  1, 2)), dim=1)
    grad_st[:,:,:,1] = torch.sum(torch.abs(image - torch.roll(image, -1, 2)), dim=1)
    grad_st[:,:,:,2] = torch.sum(torch.abs(image - torch.roll(image,  1, 3)), dim=1)
    grad_st[:,:,:,3] = torch.sum(torch.abs(image - torch.roll(image, -1, 3)), dim=1)
    grad_st[:,:,:,4] = torch.sum(torch.abs(image - torch.roll(image,  ( 1, 1), (2,3))), dim=1)
    grad_st[:,:,:,5] = torch.sum(torch.abs(image - torch.roll(image,  ( 1,-1), (2,3))), dim=1)
    grad_st[:,:,:,6] = torch.sum(torch.abs(image - torch.roll(image,  (-1, 1), (2,3))), dim=1)
    grad_st[:,:,:,7] = torch.sum(torch.abs(image - torch.roll(image,  (-1,-1), (2,3))), dim=1)
    
    grad_st = grad_st / 255
    
    # find out where there are class boundaries
    labels = torch.argmax(output, 1)
    label_diffs = torch.zeros((image.shape[0], image.shape[2], image.shape[3], 8))
    label_diffs[:,:,:,0] = labels == torch.roll(labels,  1, 1)
    label_diffs[:,:,:,1] = labels == torch.roll(labels, -1, 1)
    label_diffs[:,:,:,2] = labels == torch.roll(labels,  1, 2)
    label_diffs[:,:,:,3] = labels == torch.roll(labels, -1, 2)
    label_diffs[:,:,:,4] = labels == torch.roll(labels,  ( 1, 1), (1,2))
    label_diffs[:,:,:,5] = labels == torch.roll(labels,  ( 1,-1), (1,2))
    label_diffs[:,:,:,6] = labels == torch.roll(labels,  (-1, 1), (1,2))
    label_diffs[:,:,:,7] = labels == torch.roll(labels,  (-1,-1), (1,2))
    
    # no edge penalty at the edges!!
    label_diffs[:, 0, :,:] = 0
    label_diffs[:,-1, :,:] = 0
    label_diffs[:, :, 0,:] = 0
    label_diffs[:, :,-1,:] = 0
    
    edge_pen = grad_st * label_diffs
    num_el = image.shape[0]*image.shape[1]*image.shape[2]*image.shape[3]
    return beta*torch.sum(torch.exp(-1 * torch.mul(edge_pen, edge_pen)/K))/num_el