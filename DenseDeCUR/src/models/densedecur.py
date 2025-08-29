import torch
import torch.nn as nn
import torchvision

from .densecl import DenseCL

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    # Reshape to skip diagonal elements efficiently
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


''' DenseDeCUR '''
class DenseDeCUR(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
    
        self.mod1 = DenseCL(pretrained=True)  
        self.mod2 = DenseCL(pretrained=True)  

        self.bn = nn.BatchNorm1d(128, affine=False)

    def bt_loss_cross(self, z1, z2):
        # Compute normalized cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size*1)
        torch.distributed.all_reduce(c)

        # Split embedding space into common and unique parts
        dim_c = self.args.dim_common    # e.g. first 448 dims = common
        c_c = c[:dim_c,:dim_c]          # common-common block
        c_u = c[dim_c:,dim_c:]          # unique-unique block

        # Barlow Twins loss on common: force correlation matrix ≈ identity
        on_diag_c = torch.diagonal(c_c).add_(-1).pow_(2).sum()  # (1 - diag)^2c
        # → ∑_{i=1}^{dim_c} (c_c[i,i] - 1)^2
        off_diag_c = off_diagonal(c_c).pow_(2).sum()            # off-diag^2
        # → ∑_{i≠j} c_c[i,j]^2

        # On unique part: push values toward 0 (decorrelation only)
        on_diag_u = torch.diagonal(c_u).pow_(2).sum()
        # → ∑_{i=1}^{dim_u} c_u[i,i]^2  
        off_diag_u = off_diagonal(c_u).pow_(2).sum()
        # → ∑_{i≠j} c_u[i,j]^2
        
        # Weighted sum of on/off-diagonal penalties
        loss_c = on_diag_c + self.args.lambd * off_diag_c
        loss_u = on_diag_u + self.args.lambd * off_diag_u
        
        return loss_c,on_diag_c,off_diag_c,loss_u,on_diag_u,off_diag_u

    def forward(self, y1_1,y1_2,y2_1,y2_2):

        single_loss1, dense_loss1, z1 = self.mod1(y1_1, y1_2)
        single_loss2, dense_loss2, z2 = self.mod2(y2_1, y2_2)

        loss1 = single_loss1 + dense_loss1
        loss2 = single_loss2 + dense_loss2
        
        loss12_c, on_diag12_c, _, loss12_u, _, _ = self.bt_loss_cross(z1,z2)
        loss12 = (loss12_c + loss12_u) / 2.0

        return loss1, loss2, loss12, on_diag12_c


