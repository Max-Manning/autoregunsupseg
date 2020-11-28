import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-10

def MI_loss(out_o1, out_o2, T=10): 
    '''
    MI loss. Adapted from IID_segmentation_loss:
    https://github.com/xu-ji/IIC/blob/master/code/utils/segmentation/IID_losses.py
    
    Args:
        out_o1: output of the network using the first ordering
        out_o2: output of the network using the second ordering
        T: half window width for spatial invariance 
    '''
    B, K, H, W = out_o1.shape
    TT = 2*T + 1
#     out_o2 = random_translation_multiple(out_o2, 0, T)
    
    out_o1 = out_o1.permute(1,0,2,3)
    out_o2 = out_o2.permute(1,0,2,3)
    
    # sum over everything except classes to get the joint probability distribution
    # over the classes
    p_i_j = F.conv2d(out_o1, weight=out_o2, padding=(T,T))
    p_i_j = p_i_j.sum(dim=2, keepdim=False).sum(dim=2, keepdim=False)
    
    # normalize
#     current_norm = float(p_i_j.sum())
#     p_i_j = p_i_j / current_norm
    
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
    loss = (-p_i_j * (torch.log(p_i_j) - torch.log(p_i_mat) - 
                      torch.log(p_j_mat))).sum() / (B * TT * TT)
    
    return loss

def MI_loss_2(out_o1, out_o2, T=10): 
    '''
    MI loss. Adapted from IID_segmentation_loss_uncollapsed:
    https://github.com/xu-ji/IIC/blob/master/code/utils/segmentation/IID_losses.py
    
    Args:
        out_o1: output of the network using the first ordering
        out_o2: output of the network using the second ordering
        T: half window width for spatial invariance 
    '''
    bn, k, h, w = out_o1.shape
    
#     out_o2 = random_translation_multiple(out_o2, 0, T//2)
    
    out_o1 = out_o1.permute(1,0,2,3)
    out_o2 = out_o2.permute(1,0,2,3)
    
    # sum over everything except classes to get the joint probability distribution
    # over the classes
    p_i_j = F.conv2d(out_o1, weight=out_o2, padding=(T,T))
    Tbox = 2*T + 1
    
    # T x T x k x k
    p_i_j = p_i_j.permute(2, 3, 0, 1)
    p_i_j = p_i_j / p_i_j.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)  # norm
    
    # symmetrise, transpose the k x k part
    p_i_j = (p_i_j + p_i_j.permute(0, 1, 3, 2)) / 2.0
    
    # T x T x k x k
    p_i_mat = p_i_j.sum(dim=2, keepdim=True).repeat(1, 1, k, 1)
    p_j_mat = p_i_j.sum(dim=3, keepdim=True).repeat(1, 1, 1, k)
    
    # for log stability; tiny values cancelled out by mult with p_i_j anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_i_mat[(p_i_mat < EPS).data] = EPS
    p_j_mat[(p_j_mat < EPS).data] = EPS
    
    lamb=1.0
    
    # maximise information
    loss = (-p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_i_mat) -
        lamb * torch.log(p_j_mat))).sum() / (Tbox * Tbox)
    
    return loss


def random_translation_multiple(data, half_side_min, half_side_max):
    n, c, h, w = data.shape

    # pad last 2, i.e. spatial, dimensions, equally in all directions
    data = F.pad(data,
               (half_side_max, half_side_max, half_side_max, half_side_max),
               "constant", 0)
    assert (data.shape[2:] == (2 * half_side_max + h, 2 * half_side_max + w))

    # random x, y displacement
    t = np.random.randint(half_side_min, half_side_max + 1, size=(2,))
    polarities = np.random.choice([-1, 1], size=(2,), replace=True)
    t *= polarities

    # -x, -y in orig img frame is now -x+half_side_max, -y+half_side_max in new
    t += half_side_max

    data = data[:, :, t[1]:(t[1] + h), t[0]:(t[0] + w)]
    assert (data.shape[2:] == (h, w))

    return data