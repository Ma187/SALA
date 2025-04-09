from abc import ABC
from typing import Dict, Optional

import math
from src.models.PatchTST_layers import *
from src.models.model import *


class MILLET(nn.Module):
    def __init__(self, feature_extractor, pool=None, args=None, pe='zeros', learn_pe=True, patch_dim=32, dropout=0.,
                 decoder=None):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        self.pool = pool
        self.args = args
        self.patch_len = args.patch_len
        self.shapelet_len = max(int(self.args.len_shapelet[0]*self.args.sample_len), self.patch_len)
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

        interpre_len = (math.floor((self.args.sample_len-self.shapelet_len)/self.stride) + 1) if args.patch else int(args.sample_len)
        padding = max(int(((interpre_len-1)*average_stride+self.patch_stride-self.args.sample_len)/2), 0)

        self.averagePool = nn.AvgPool1d(kernel_size=average_size,
                                        stride=average_stride,
                                        padding=padding)

        if args.interpre_type == 'atten':
            self.interpre = torch.zeros(args.num_training_samples, interpre_len, 1)
        else: # interpre_type == 'interpretation'
            self.interpre = torch.zeros(args.num_training_samples, args.nbins, interpre_len)

        self.W_P = None
        self.W_pos = None
        if self.args.patch and self.args.patch_type == 'before_encode':
            self.W_P = nn.Linear(self.patch_len, patch_dim)
            final_len = (math.floor((self.args.sample_len - self.patch_len) / self.patch_stride) + 1) if args.patch else int(
                args.sample_len)
            self.W_pos = positional_encoding(pe, learn_pe, final_len, patch_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, bags, pos=None, recon=False, conf_mask=None, label=None, forward_type='None',
                conf_score=None,x_idx=None,only_max_min='max_min'):

        if self.args.patch and self.args.patch_type == 'before_encode':
            x = bags.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
              # x: [bs x nvars x patch_num x patch_len]
            x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]
            x = x.squeeze(1)
            bags = self.dropout(x + self.W_pos).transpose(1,2)
        timestep_embeddings = self.feature_extractor(bags)
        if recon:
            x_recon = self.decoder(timestep_embeddings)
        if self.args.patch and self.args.patch_type == 'after_encode':
            if self.args.sample_len <= self.patch_len:
                timestep_embeddings = timestep_embeddings.mean(dim=-1).unsqueeze(-1)
            else:
                timestep_embeddings = self.averagePool(timestep_embeddings)
        # elif self.args.patch and self.args.patch_type == 'before_encode': # 尝试patch长度和shapelet长度解耦
        #     timestep_embeddings = self.averagePool(timestep_embeddings)

        bag_out = self.pool(timestep_embeddings, pos=pos, conf_mask=conf_mask,obslabel=label)

        if recon:
            return bag_out, x_recon
        else:
            return bag_out




class MILPooling(nn.Module, ABC):
    """Base class for MIL pooling methods."""

    def __init__(
            self,
            d_in: int,
            n_clz: int,
            dropout: float = 0.1,
            apply_positional_encoding: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_clz = n_clz
        self.dropout_p = dropout
        self.apply_positional_encoding = apply_positional_encoding
        # Create positional encoding and dropout layers if needed
        if apply_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_in)
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)


class MILConjunctivePooling(MILPooling):
    """Conjunctive MIL pooling. Instance attention then weighting of instance predictions."""

    def __init__(
            self,
            d_in: int,
            n_clz: int,
            d_attn: int = 8,
            dropout: float = 0.1,
            apply_positional_encoding: bool = True,
            classifier=None,
            args=None
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        if classifier is None:
            self.instance_classifier = nn.Linear(d_in, n_clz)
        else:
            self.instance_classifier = classifier
        # self.weak = args.weak if args else 0.5

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None, s_trans=None,
                conf_mask = None, obslabel = None, smoothy_instance_logits=None, args=None) -> Dict[str, torch.Tensor]:
        """
        :param instance_embeddings: Torch tensor of shape: batch x n_channels x n_timesteps
        :param pos: Optional torch tensor of position (index) for each instance; shape: n_timesteps
        :param conf_mask: used in training.
        :param obslabel: observed labels of training data
        :return: Dictionary containing bag_logits, interpretation (instance predictions weight by attention),
        unweighted instance logits, and attn values.
        """

        # Swap instance embeddings to batch x n_timesteps x n_channels
        #  Equivalent to batch x n_instances x n_channels
        instance_embeddings = instance_embeddings.transpose(1, 2)
        # Add positional encodings
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        # Apply dropout
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        # Calculate attention
        attn = self.attention_head(instance_embeddings)
        # Classify instances
        if s_trans is not None:
            # instance_embeddings = torch.cat((instance_embeddings, s_trans.repeat(1,instance_embeddings.shape[1],1)),dim=2)
            instance_embeddings = instance_embeddings + s_trans.repeat(1,instance_embeddings.shape[1],1)
        if conf_mask is None:
            instance_logits = self.instance_classifier(instance_embeddings)
            weighted_instance_logits = instance_logits * attn
            if smoothy_instance_logits is not None: # not None in training
                weighted_instance_logits = args.smooth_beta * weighted_instance_logits + \
                                           (1-args.smooth_beta) * smoothy_instance_logits.transpose(1,2).cuda()
        # Weight and sum
            bag_logits = torch.mean(weighted_instance_logits, dim=1)
        else:
            instance_logits, weighted_instance_logits, bag_logits = \
                self.getBagLogit(instance_embeddings,attn,conf_mask,obslabel)
        return {
            "bag_logits": bag_logits,
            "interpretation": weighted_instance_logits.transpose(1, 2),
            # Also return additional outputs
            "instance_logits": instance_logits.transpose(1, 2),
            "atten": attn,
        }

    def getBagLogit(self, instance_embeddings, attn, conf_mask, obsLabel, weak=0.5):
        # training stage
        conf_mask = ~conf_mask
        instance_logits = self.instance_classifier(instance_embeddings)

        modified_attn = attn.clone()

        with torch.no_grad():
            tmp_instance_logits = instance_logits * modified_attn
            score, plabel = tmp_instance_logits.max(axis=-1)

            conf_mask_indices = torch.where(conf_mask == 1)[0]

            for idx in conf_mask_indices:
                matching_instances_indices = torch.where(plabel[idx] == obsLabel[idx])[0]
                modified_attn[idx, matching_instances_indices] = modified_attn[idx, matching_instances_indices] * weak

        weighted_instance_logits = instance_logits * modified_attn
        # Weight and sum
        bag_logits = torch.mean(weighted_instance_logits, dim=1)
        return instance_logits, weighted_instance_logits, bag_logits


class PositionalEncoding(nn.Module):
    """
    Adapted from (under BSD 3-Clause License):
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Batch, ts len, d_model
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor, x_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply positional encoding to a set of time series embeddings.

        :param x: Embeddings.
        :param x_pos: Optional positions (indices) of each timestep. If not provided, will use range(len(time series)),
        i.e. 0,...,t-1.
        :return: A tensor the same shape as x, but with positional encoding added to it.
        """
        if x_pos is None:
            x_pe = self.pe[:, : x.size(1)]
        else:
            x_pe = self.pe[0, x_pos]
        x = x + x_pe
        return x


class GlobalAveragePooling(MILPooling):
    """GAP (EmbeddingSpace MIL) pooling."""

    def __init__(
            self,
            d_in: int,
            n_clz: int,
            dropout: float = 0,
            apply_positional_encoding: bool = False,
            classifier=None,
            args=None
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )
        # if classifier is None:
        self.bag_classifier = nn.Linear(d_in, n_clz)
        # else:
        #     self.bag_classifier = classifier

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None, s_trans=None,
                conf_mask = None, obslabel = None) -> Dict[str, torch.Tensor]:
        """
        :param instance_embeddings: Torch tensor of shape: batch x n_channels x n_timesteps
        :param pos: Optional torch tensor of position (index) for each instance; shape: n_timesteps
        :return: Dictionary containing bag_logits and interpretation (CAM).
        """
        # Swap instance embeddings to batch x n_timesteps x n_channels
        #  Equivalent to batch x n_instances x n_channels
        instance_embeddings = instance_embeddings.transpose(1, 2)
        # Add positional encodings
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        # Apply dropout
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        # Swap instance embeddings back after adding positional embeddings (if using). Needed for CAM
        instance_embeddings = instance_embeddings.transpose(2, 1)
        # Calculate class activation map (CAM)
        cam = self.bag_classifier.weight @ instance_embeddings
        # Mean pool (GAP) to bag embeddings
        bag_embeddings = instance_embeddings.mean(dim=-1)
        # Classify the bag embeddings
        bag_logits = self.bag_classifier(bag_embeddings)
        return {
            "bag_logits": bag_logits,
            "interpretation": cam,
        }



class MILInstancePooling(MILPooling):
    """Instance MIL pooling. Instance prediction then averaging."""

    def __init__(
            self,
            d_in: int,
            n_clz: int,
            dropout: float = 0.1,
            apply_positional_encoding: bool = True,
            classifier=None
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding
        )
        if classifier is None:
            self.instance_classifier = nn.Linear(d_in, n_clz)
        else:
            self.instance_classifier = classifier

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        :param instance_embeddings: Torch tensor of shape: batch x n_channels x n_timesteps
        :param pos: Optional torch tensor of position (index) for each instance; shape: n_timesteps
        :return: Dictionary containing bag_logits and interpretation (instance predictions).
        """
        # Swap instance embeddings to batch x n_timesteps x n_channels
        #  Equivalent to batch x n_instances x n_channels
        instance_embeddings = instance_embeddings.transpose(1, 2)
        # Add positional encodings
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        # Apply dropout
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        # Classify instances
        instance_logits = self.instance_classifier(instance_embeddings)
        # Mean pool to bag prediction
        bag_logits = instance_logits.mean(dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": instance_logits.transpose(1, 2),
        }


class MILAttentionPooling(MILPooling):
    """Attention MIL pooling. Instance attention then weighted averaging of embeddings."""

    def __init__(
            self,
            d_in: int,
            n_clz: int,
            d_attn: int = 8,
            dropout: float = 0.1,
            apply_positional_encoding: bool = True,
            classifier=None
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        if classifier is None:
            self.bag_classifier = nn.Linear(d_in, n_clz)
        else:
            self.bag_classifier = classifier

    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        :param instance_embeddings: Torch tensor of shape: batch x n_channels x n_timesteps
        :param pos: Optional torch tensor of position (index) for each instance; shape: n_timesteps
        :return: Dictionary containing bag_logits and interpretation (attention).
        """
        # Swap instance embeddings to batch x n_timesteps x n_channels
        #  Equivalent to batch x n_instances x n_channels
        instance_embeddings = instance_embeddings.transpose(1, 2)
        # Add positional encodings
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        # Apply dropout
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        # Calculate attention
        attn = self.attention_head(instance_embeddings)
        # Use attention to get bag embedding
        instance_embeddings = instance_embeddings * attn
        bag_embedding = torch.mean(instance_embeddings, dim=1)
        # Classify the bag embedding
        bag_logits = self.bag_classifier(bag_embedding)
        return {
            "bag_logits": bag_logits,
            # Attention is not class wise, so repeat for each class
            "interpretation": attn.repeat(1, 1, self.n_clz).transpose(1, 2),
        }


class MILAdditivePooling(MILPooling):
    """Additive MIL pooling. Instance attention then weighting of embeddings before instance prediction."""

    def __init__(
            self,
            d_in: int,
            n_clz: int,
            d_attn: int = 8,
            dropout: float = 0.1,
            apply_positional_encoding: bool = True,
            classifier=None
    ):
        super().__init__(
            d_in,
            n_clz,
            dropout=dropout,
            apply_positional_encoding=apply_positional_encoding,
        )
        self.attention_head = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
            nn.Sigmoid(),
        )
        if classifier is None:
            self.instance_classifier = nn.Linear(d_in, n_clz)
        else:
            self.instance_classifier = classifier


    def forward(self, instance_embeddings: torch.Tensor, pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        :param instance_embeddings: Torch tensor of shape: batch x n_channels x n_timesteps
        :param pos: Optional torch tensor of position (index) for each instance; shape: n_timesteps
        :return: Dictionary containing bag_logits, interpretation (instance predictions weight by attention),
        unweighted instance logits, and attn values.
        """
        # Swap instance embeddings to batch x n_timesteps x n_channels
        #  Equivalent to batch x n_instances x n_channels
        instance_embeddings = instance_embeddings.transpose(1, 2)
        # Add positional encodings
        if self.apply_positional_encoding:
            instance_embeddings = self.positional_encoding(instance_embeddings, pos)
        # Apply dropout
        if self.dropout_p > 0:
            instance_embeddings = self.dropout(instance_embeddings)
        # Calculate attention
        attn = self.attention_head(instance_embeddings)
        # Scale instances and classify
        instance_embeddings = instance_embeddings * attn
        instance_logits = self.instance_classifier(instance_embeddings)
        # Mean pool to bag prediction
        bag_logits = instance_logits.mean(dim=1)
        return {
            "bag_logits": bag_logits,
            "interpretation": (instance_logits * attn).transpose(1, 2),
            # Also return additional outputs
            "instance_logits": instance_logits.transpose(1, 2),
            "atten": attn,
        }



