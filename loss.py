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