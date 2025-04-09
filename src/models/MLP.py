import torch
from torch import Tensor
from torch import nn

class MLPEncoder(nn.Module):
    def __init__(self, c_in, patch_len, d_model=128, shared_embedding=True):
        super().__init__()
        self.n_vars = c_in
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding
        self.act = nn.ReLU(inplace=True)
        if not shared_embedding:
            self.W_P2 = nn.ModuleList()
            for _ in range(self.n_vars):
                self.W_P2.append(nn.Linear(d_model, d_model))
        else:
            self.W_P2 = nn.Linear(d_model, d_model)

    def forward(self, x) -> Tensor:
        """
        x: tensor [bs x n_vars x dmodel x num_patch]
        """
        bs, n_vars, dmodel, num_patch = x.shape
        # Input encoding
        x = x.transpose(2, 3)                   # [bs x n_vars x num_patch x dmodel]
        if not self.shared_embedding:
            x_out2 = []
            for i in range(n_vars):
                z = self.W_P2[i](x)
                x_out2.append(z)
            x2 = torch.stack(x_out2, dim=1)
        else:
            x2 = self.W_P2(x)  # x: [bs x n_vars x num_patch x dmodel]
        x2 = x2.permute(0, 1, 3, 2)
        return x2

class MLPDecoder(nn.Module):
    def __init__(self, d_model, patch_len, dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        x = x.transpose(2,3)                     # [bs x nvars x num_patch x d_model]
        x = self.linear( self.dropout(x) )      # [bs x nvars x num_patch x patch_len]
        return x