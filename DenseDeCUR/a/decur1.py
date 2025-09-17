import torch
import torch.nn as nn
import torchvision



def off_diagonal(x):
    """
    Returns all off-diagonal elements of a square matrix as a flat tensor.
    Args:
        x (torch.Tensor): Square matrix of shape (n, n).
    Returns:
        torch.Tensor: Flattened off-diagonal elements.
    """
    n, m = x.shape
    assert n == m
    # Reshape to skip diagonal elements efficiently
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

''' DeCUR '''
class DeCUR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # backbone
        if args.backbone == 'resnet50':
            self.backbone_1 = torchvision.models.resnet50(zero_init_residual=True,pretrained=True)
            self.backbone_2 = torchvision.models.resnet50(zero_init_residual=True,pretrained=True)

        # Backbone adaptation (input channels & head removal)
        self.backbone_1.fc = nn.Identity()
        self.backbone_2.fc = nn.Identity()


        # projector: Build projector MLPs for each backbone output
        if args.backbone == 'resnet50':
            sizes = [2048] + list(map(int, args.projector.split('-'))) # e.g. sizes = [2048, 8192, 8192, 8192]

        # Build MLP with linear layers + BatchNorm + ReLU (except last layer)
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        # Final linear layer without activation
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))

        # Define two separate projectors (one for each modality/view)
        self.projector1 = nn.Sequential(*layers)
        self.projector2 = nn.Sequential(*layers)

        # Final normalization layer for the representations z1 and z2:
        # affine=False → no learnable parameters, just statistical normalization
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def bt_loss_cross(self, z1, z2):
        """
        Compute cross-view Barlow Twins loss by splitting the embedding into 
        common and unique components, and decorrelating them accordingly.

        Args:
            z1 (Tensor): Projected embeddings from view 1, shape (B, D)
            z2 (Tensor): Projected embeddings from view 2, shape (B, D)

        Returns:
            loss_c (Tensor): Loss on common components
            on_diag_c (Tensor): Diagonal penalty (common)
            off_diag_c (Tensor): Off-diagonal penalty (common)
            loss_u (Tensor): Loss on unique components
            on_diag_u (Tensor): Diagonal penalty (unique)
            off_diag_u (Tensor): Off-diagonal penalty (unique)
        """
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


    def bt_loss_single(self, z1, z2):
        """
        Compute Barlow Twins loss between two augmentations of the same view.
        Args:
            z1 (Tensor): Projected embeddings from augmentation 1, shape (B, D)
            z2 (Tensor): Projected embeddings from augmentation 2, shape (B, D)
        Returns:
            loss (Tensor): Total Barlow Twins loss
            on_diag (Tensor): Diagonal penalty term
            off_diag (Tensor): Off-diagonal penalty term
        """
        # Compute normalized cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # Normalize by total number of samples (assumes 4 GPUs)
        c.div_(self.args.batch_size*1)
        torch.distributed.all_reduce(c)

        # Diagonal term: (1 - c_ii)^2
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()  # → ∑_{i=1}^{D} (c[i,i] - 1)^2
       
        # Off-diagonal term: c_ij^2 for i ≠ j
        off_diag = off_diagonal(c).pow_(2).sum()            # → ∑_{i≠j} c[i,j]^2
       
        # Final loss: diagonal + λ * off-diagonal
        loss = on_diag + self.args.lambd * off_diag         # → L = ∑_i (c_ii - 1)^2 + λ ∑_{i≠j} c_ij^2
        
        return loss,on_diag,off_diag


    def forward(self, y1_1,y1_2,y2_1,y2_2):
        """
        Forward pass for contrastive self-supervised training.

        Args:
            y1_1, y1_2: Two augmentations of input modality 1 (e.g. Sentinel-1)
            y2_1, y2_2: Two augmentations of input modality 2 (e.g. Sentinel-2)

        Returns:
            loss1 (Tensor): Barlow Twins loss for view 1 (intra-modal)
            loss2 (Tensor): Barlow Twins loss for view 2 (intra-modal)
            loss12 (Tensor): Cross-view loss (inter-modal)
            on_diag12_c (Tensor): Diagonal loss term for shared representation
        """

        # Extract features with backbone 
        f1_1 = self.backbone_1(y1_1)
        f1_2 = self.backbone_1(y1_2)
        f2_1 = self.backbone_2(y2_1)
        f2_2 = self.backbone_2(y2_2)  

        # Project features into embedding space
        z1_1 = self.projector1(f1_1)
        z1_2 = self.projector1(f1_2)
        z2_1 = self.projector2(f2_1)
        z2_2 = self.projector2(f2_2)         

        # Intra-view losses (Barlow Twins): same modality, different augmentations
        loss1, on_diag1, off_diag1 = self.bt_loss_single(z1_1,z1_2)
        loss2, on_diag2, off_diag2 = self.bt_loss_single(z2_1,z2_2)        
        
        # Inter-view loss: cross-modal alignment and separation
        loss12_c, on_diag12_c, off_diag12_c, loss12_u, on_diag12_u, off_diag12_u = self.bt_loss_cross(z1_1,z2_1)
        loss12 = (loss12_c + loss12_u) / 2.0

        return loss1,loss2,loss12,on_diag12_c