import torch
import torch.nn as nn
import torchvision

from densecl import DenseCL
from necks   import densecl_neck
from resnet  import resnet50


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


''' DenseDeCUR '''
class DenseDeCUR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # backbones
        self.b1_q = resnet50()
        self.b1_k = resnet50()
        self.b2_q = resnet50()
        self.b2_k = resnet50()

        # per KAIST (modificare qui )
        #if args.mode==['s1','s2c']:
        #    self.b1_q[0] = torch.nn.Conv2d(2 , 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
        #    self.b1_k[0] = torch.nn.Conv2d(2 , 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
        #    self.b2_q[0] = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        #    self.b2_k[0] = torch.nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # neck
        self.n1_q = densecl_neck(in_channels=2048, hid_channels=2048, out_channels=128)
        self.n1_k = densecl_neck(in_channels=2048, hid_channels=2048, out_channels=128)
        self.n2_q = densecl_neck(in_channels=2048, hid_channels=2048, out_channels=128)
        self.n2_k = densecl_neck(in_channels=2048, hid_channels=2048, out_channels=128)

        # modality 
        self.densecl_1 = DenseCL(self.b1_q, self.n1_q, self.b1_k, self.n1_k, pretrained=True)
        self.densecl_2 = DenseCL(self.b2_q, self.n2_q, self.b2_k, self.n2_k, pretrained=True)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(128, affine=False) 

    def bt_loss_cross(self, z1, z2):
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size*4)
        torch.distributed.all_reduce(c)

        dim_c = self.args.dim_common
        c_c = c[:dim_c,:dim_c]
        c_u = c[dim_c:,dim_c:]

        on_diag_c = torch.diagonal(c_c).add_(-1).pow_(2).sum()
        off_diag_c = off_diagonal(c_c).pow_(2).sum()
        
        on_diag_u = torch.diagonal(c_u).pow_(2).sum()
        off_diag_u = off_diagonal(c_u).pow_(2).sum()
        
        loss_c = on_diag_c + self.args.lambd * off_diag_c
        loss_u = on_diag_u + self.args.lambd * off_diag_u

        loss = (loss_c + loss_u) / 2.0
        
        return loss


    def forward(self, y1_1,y1_2,y2_1,y2_2):

        out1 = self.densecl_1(y1_1, y1_2)
        out2 = self.densecl_2(y2_1, y2_2)
        
        # loss intra-modale con DenseCL
        loss1 = out1["loss_intra"]
        loss2 = out2["loss_intra"]

        # loss cross-modale
        z1 = out1["zq_global"]
        z2 = out2["zq_global"]
        loss12 = self.bt_loss_cross(z1,z2)

        return loss1, loss2, loss12 
    

