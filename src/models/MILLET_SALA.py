import random
import math
import torch
from src.models.PatchTST_layers import *
from src.models.model import *


class MILLET(nn.Module):
    def __init__(self, feature_extractor, pool=None, args=None, pe='zeros', learn_pe=True, patch_dim=32, dropout=0.,
                 decoder=None):
        super().__init__()

        self.amp_noise = args.amp_noise
        self.amp_mask = args.amp_mask
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        self.pool = pool
        self.args = args
        self.patch_len = args.patch_len
        self.mean_norm = args.mean_norm
        self.act = nn.ReLU(inplace=True)
        self.shapelet_len = self.patch_len
        self.stride = int(self.args.shapelet_stride)
        if args.patch and args.patch_type == 'before_encode':
            self.shapelet_len = self.shapelet_len - self.shapelet_len % self.patch_len
        average_size = self.shapelet_len
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
        if self.args.patch and self.args.patch_type == 'before_encode':
            self.W_P = nn.Linear(self.patch_len*self.args.nvars, patch_dim)
            final_len = (math.floor(
                (self.args.sample_len - self.patch_len) / self.patch_stride) + 1) if args.patch else int(
                args.sample_len)
            self.W_pos = positional_encoding(pe, learn_pe, final_len, patch_dim)
        self.dropout = nn.Dropout(dropout)

    def addNoiseInNoisyPatchEmb(self, patch, labels, conf_score, x_idx, only_max_min,conf_mask=None):
        # patch: [bs x nvars x patch_num x d_model]
        with torch.no_grad():
            interpre = self.interpre[x_idx].cuda()
            interpre = torch.softmax(interpre, dim=1)

            # noise = torch.normal(0, 1, patch.shape)
            variances = torch.var(patch, dim=-1, keepdim=True)
            noise = torch.randn_like(patch) * variances.sqrt()
            noise = noise.cuda()

            noise_prob = (1 - conf_score[x_idx])
            noise_prob = noise_prob.view(patch.shape[0], 1, 1, 1)
            conf_prob = conf_score[x_idx].view(patch.shape[0], 1, 1, 1)
            noise = noise * self.amp_noise

            labels = labels[:, None, None].expand(-1, -1, interpre.shape[2]).cuda()
            instance_score = torch.gather(interpre, 1, labels)
            if only_max_min in ['max_min','max','near_max_min']:
                noise_ = torch.zeros_like(noise)
                if only_max_min == 'near_max_min':
                    m = (instance_score > torch.max(instance_score)-self.args.score_gap).float()
                    m = m[:,:,:,None].repeat(1,1,1,patch.shape[3])
                    noise_ = noise*noise_prob*m
                else:
                    indices1 = torch.argmax(instance_score, dim=-1, keepdim=True).unsqueeze(2).repeat(1,1,1,noise.shape[3])
                    select_noise1 = torch.gather(noise, 2, indices1)
                    noise_.scatter_(2, indices1, (select_noise1*noise_prob).float())
                if only_max_min in ['max_min','near_max_min']:
                    if only_max_min == 'near_max_min':
                        m = (instance_score < torch.min(instance_score) + self.args.score_gap).float()
                        m = m[:, :, :, None].repeat(1, 1, 1, patch.shape[3])
                        noise_ += noise * conf_prob * m
                    else:
                        indices2 = torch.argmin(instance_score, dim=-1, keepdim=True).unsqueeze(2).repeat(1,1,1,noise.shape[3])
                        select_noise2 = torch.gather(noise, 2, indices2)
                        noise_.scatter_(2, indices2, (select_noise2 * conf_prob).float())
                patch = patch + noise_.cuda()
            else:
                noise = (noise * noise_prob).cuda()
                patch = patch + noise * instance_score.unsqueeze(-1).cuda()
        return patch

    def maskNoisyPatch(self, x, labels, conf_score, x_idx, only_max_min,conf_mask=None):
        # x: [bs x nvars x patch_num x patch_len]
        with torch.no_grad():
            noise_prob = 1 - conf_score[x_idx]
            noise_prob = noise_prob.view(x.shape[0], 1, 1, 1)
            conf_prob = conf_score[x_idx].view(x.shape[0], 1, 1, 1)

            interpre = self.interpre[x_idx].cuda()
            interpre = torch.softmax(interpre, dim=1)
            labels = labels[:, None, None].expand(-1, -1, interpre.shape[2]).cuda()
            instance_score = torch.gather(interpre, 1, labels)

            noise_prob = noise_prob.expand_as(x)
            conf_prob = conf_prob.expand_as(x)
            if only_max_min in ['max_min','max','near_max_min']:
                mask_ratio = torch.zeros_like(x).cuda()
                if only_max_min == 'near_max_min':
                    m = (instance_score > torch.max(instance_score)-self.args.score_gap).float()
                    m = m[:,:,:,None].repeat(1,1,1,x.shape[3])
                    mask_ratio = noise_prob*m
                else:
                    indices1 = torch.argmax(instance_score, dim=-1, keepdim=True).unsqueeze(3).repeat(1,1,1,x.shape[3])
                    select_mask_ratio1 = torch.gather(noise_prob, 2, indices1)
                    mask_ratio.scatter_(2, indices1, select_mask_ratio1.float())
                if only_max_min in ['max_min','near_max_min']:
                    if only_max_min == 'near_max_min':
                        m = (instance_score < torch.min(instance_score) + self.args.score_gap).float()
                        m = m[:,:,:,None].repeat(1,1,1,x.shape[3])
                        mask_ratio += noise_prob * m
                    else:
                        indices2 = torch.argmin(instance_score, dim=-1, keepdim=True).unsqueeze(3).repeat(1,1,1,x.shape[3])
                        select_mask_ratio2 = torch.gather(conf_prob, 2, indices2)
                        mask_ratio.scatter_(2, indices2, select_mask_ratio2.float())

            else:
                mask_ratio = noise_prob.mul(instance_score.unsqueeze(-1).expand_as(x))

            random_mask = (torch.rand_like(x).cuda() < self.amp_mask*mask_ratio).float()
            x[conf_mask] = x[conf_mask] * (1 - random_mask)[conf_mask]

        return x

    def maskNoisyPatch_sanm(self, x, labels, conf_score, x_idx, only_max_min,conf_mask=None):
        # x: [bs x nvars x patch_num x patch_len]
        with torch.no_grad():
            length = x.shape[-2]
            noise_prob = 1 - conf_score[x_idx]
            noise_prob = noise_prob.view(-1)
            conf_prob = conf_score[x_idx].view(-1)

            interpre = self.interpre[x_idx].cuda()
            interpre = torch.softmax(interpre, dim=1)
            labels = labels[:, None, None].expand(-1, -1, interpre.shape[2]).cuda()
            instance_score = torch.gather(interpre, 1, labels).view(x.shape[0],-1)

            r1 = 0.8
            indices1 = torch.argmax(instance_score, dim=-1, keepdim=True).view(x.shape[0])
            indices2 = torch.argmin(instance_score, dim=-1, keepdim=True).view(x.shape[0])


            for i in range(x.shape[0]):
                for attempt in range(100):

                    aspect_ratio = random.uniform(r1, 1 / r1)
                    ll = int(round( 0.5 *length * noise_prob[i].numpy() * aspect_ratio))
                    if ll==0: break
                    if ll < length:
                        x1 = indices1[i]
                        x1r = x1+ll
                        if x1r>length:
                            x1r=length
                        x1l = x1-ll
                        if x1l<0:
                            x1l=0
                        x[i, 0,x1l:x1r,:] = random.uniform(0, 1)


                    ll = int(round(0.5 * length * conf_prob[i].numpy() * aspect_ratio))
                    if ll==0: break
                    if ll < length:
                        x2 = indices2[i]
                        x2r = x2+ll
                        if x2r>length:
                            x2r=length
                        x2l = x2-ll
                        if x2l<0:
                            x2l=0

                        x[i,0, x2l:x2r,:] = random.uniform(0, 1)
                        break
        return x

    def forward(self, bags, pos=None, recon=False, conf_mask=None, label=None, forward_type='None',
                conf_score=None, x_idx=None, only_max_min_noise='max_min', only_max_min_mask='max_min',
                mode='None'):
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
        if forward_type == 'mask' or self.args.model == 'sanm':
            predLabel = self.predLabel[x_idx]
            if self.args.perturbPred:
                if self.args.model == 'sanm':
                    x = self.maskNoisyPatch_sanm(x, predLabel, conf_score, x_idx, only_max_min_mask, conf_mask)
                else:
                    x = self.maskNoisyPatch(x, predLabel, conf_score, x_idx, only_max_min_mask, conf_mask)
            else:
                x = self.maskNoisyPatch(x, label, conf_score, x_idx, only_max_min_mask,conf_mask)

        if self.mean_norm:
            x_mean = x.mean(axis=2).mean(axis=-1).unsqueeze(2).unsqueeze(-1)
            x = x - x_mean
        x = self.W_P(x.transpose(1,2).contiguous().view(B, patch_num, -1))  # x: [bs x nvars x patch_num x d_model]
        
        if self.args.backbone == 'MLP':
            x = self.act(x)
        predLabel = self.predLabel[x_idx].long().cuda()
        if forward_type == 'perturb':
            if self.args.perturbPred:
                x = self.addNoiseInNoisyPatchEmb(x[:, None, :, :], predLabel, conf_score, x_idx, only_max_min_noise, conf_mask)
            else:
                x = self.addNoiseInNoisyPatchEmb(x[:,None,:,:], label, conf_score, x_idx, only_max_min_noise,conf_mask)

        if self.args.backbone == 'FCN':
            x = x.squeeze(1)
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

        bag_out = self.pool(timestep_embeddings.view(B,-1,num_patch), pos=pos, conf_mask=conf_mask, obslabel=label,
                            smoothy_instance_logits=self.interpre[x_idx] if mode=='smooth' else None, args=self.args)

        if recon:
            return bag_out, x_recon
        else:
            return bag_out

