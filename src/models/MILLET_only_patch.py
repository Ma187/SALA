import math
import torch
from src.models.PatchTST_layers import *
from src.models.model import *
import numpy as np


class MILLET(nn.Module):
    def __init__(self, feature_extractor, pool=None, args=None, pe='zeros', learn_pe=True, patch_dim=32, dropout=0.,
                 decoder=None,  low_dim=None):
        super().__init__()
        '''
        type:
            'train' or 'infer'. 'train' for updating the model. 
            'infer' for predict the label for add noise/mask 
        '''
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        self.pool = pool
        self.args = args
        self.patch_len = args.patch_len
        self.mean_norm = args.mean_norm
        self.act = nn.ReLU(inplace=True)
        # self.shapelet_len = max(int(self.args.len_shapelet[0] * self.args.sample_len), self.patch_len)
        self.shapelet_len = self.patch_len
        self.stride = int(self.args.shapelet_stride)
        if args.patch and args.patch_type == 'before_encode':
            self.shapelet_len = self.shapelet_len - self.shapelet_len % self.patch_len
            # self.stride = self.stride - self.stride % self.patch_len
        average_size = self.shapelet_len
        # self.patch_stride = int(args.patch_len*args.patch_stride)
        self.patch_stride = self.stride
        average_stride = self.stride

        if args.patch and args.patch_type == 'before_encode':
            average_size = self.shapelet_len // self.patch_len
            average_stride = self.stride // self.patch_len


        interpre_len = (
                math.floor((self.args.sample_len - self.shapelet_len) / self.stride) + 1) if args.patch else int(
            args.sample_len)
        padding = max(int(((interpre_len - 1) * average_stride + self.patch_stride - self.args.sample_len) / 2), 0)

        self.averagePool = nn.AvgPool1d(kernel_size=average_size,
                                        stride=average_stride,
                                        padding=padding)

        if args.interpre_type == 'atten':
            self.interpre = torch.zeros(args.num_training_samples, interpre_len, 1)
        else:  # interpre_type == 'interpretation'
            self.interpre = torch.zeros(args.num_training_samples, args.nbins, interpre_len)
        self.predLabel = torch.zeros(args.num_training_samples).long()
        self.W_P = None
        self.W_pos = None
        self.W_pos_scale2 = None
        self.W_pos_scale4 = None
        if self.args.patch and self.args.patch_type == 'before_encode':
            self.W_P = nn.Linear(self.patch_len*self.args.nvars, patch_dim)
            final_len = (math.floor(
                (self.args.sample_len - self.patch_len) / self.patch_stride) + 1) if args.patch else int(
                args.sample_len)
            final_len_scale2 = (math.floor(
                (np.ceil(self.args.sample_len/2) - self.patch_len) / self.patch_stride) + 1) if args.patch else int(
                args.sample_len)
            final_len_scale4 = (math.floor(
                (np.ceil(self.args.sample_len/4) - self.patch_len) / self.patch_stride) + 1) if args.patch else int(
                args.sample_len)
            self.W_pos = positional_encoding(pe, learn_pe, final_len, patch_dim)
            self.W_pos_scale2 = positional_encoding(pe, learn_pe, final_len_scale2, patch_dim)
            self.W_pos_scale4 = positional_encoding(pe, learn_pe, final_len_scale4, patch_dim)
        self.dropout = nn.Dropout(dropout)
        if args.model=='Sel_CL':
            self.head = nn.Linear(128, low_dim)

    def forward(self, bags, pos=None, recon=False, conf_mask=None, label=None,
                x_idx=None, encoder=False, feat_selcl=False, scale = 0):
        '''
        :param bags:
        :param pos:
        :param recon:
        :param conf_mask: `
        :param label:
        :param forward_type: perturbation, add noise in patch or mask patch; None.
        :param conf_score:
        :param interpre:
        :return:
        '''

        x = bags.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        B, nvars, patch_num, patch_len = x.shape
        # x: [bs x nvars x patch_num x patch_len]

        if self.mean_norm:
            x_mean = x.mean(axis=2).mean(axis=-1).unsqueeze(2).unsqueeze(-1)
            x = x - x_mean
        if self.args.patch:
            x = self.W_P(x.transpose(1,2).contiguous().view(B, patch_num, -1))  # x: [bs x nvars x patch_num x d_model]
        
        if self.args.backbone == 'MLP':
            x = self.act(x)

        if self.args.backbone == 'FCN':
            x = x.squeeze(1)
        if scale == 2:
            bags = self.dropout(x + self.W_pos_scale2).transpose(-2, -1).float()
        elif scale ==4:
            bags = self.dropout(x + self.W_pos_scale4).transpose(-2, -1).float()
        else:
            bags = self.dropout(x + self.W_pos).transpose(-2, -1).float()

        timestep_embeddings = self.feature_extractor(bags)

        if recon:
            x_recon = self.decoder(timestep_embeddings)
            if self.mean_norm:
                x_recon += x_mean
        if self.args.backbone == 'MLP':
            B, nvars, dmodel, num_patch = timestep_embeddings.shape
        else:
            B, dmodel, num_patch = timestep_embeddings.shape
        bag_out = self.pool(timestep_embeddings.view(B,-1,num_patch), pos=pos, conf_mask=conf_mask, obslabel=label)

        sample_embeddings = torch.mean(timestep_embeddings, dim=2)

        if feat_selcl:
            sample_embeddings = sample_embeddings.view(sample_embeddings.size(0), -1)
            outContrast = self.head(sample_embeddings)
            sample_embeddings = F.normalize(outContrast, dim=1)

        if recon and encoder:
            return bag_out, x_recon, sample_embeddings
        elif recon:
            return bag_out, x_recon
        elif encoder:
            return bag_out, sample_embeddings
        else:
            return bag_out


