import math
import torch
from src.models.PatchTST_layers import *
from src.models.model import *


def attention_erase_map(images, outputs, gmmweight, target_mask):
    erase_x=[]
    erase_y=[]
    erase_x_min=[]
    erase_y_min=[]
    width=images.shape[2]
    height=images.shape[3]
    outputs = (outputs**2).sum(1)
    b, h, w = outputs.size()#shape
    outputs = outputs.view(b, h * w)
    outputs = F.normalize(outputs, p=2, dim=1)
    outputs = outputs.view(b, h, w)
    for j in range(outputs.size(0)):
        am = outputs[j, ...].detach().cpu().numpy()
        am = cv2.resize(am, (width, height))
        am = 255 * (am - np.min(am)) / (
                    np.max(am) - np.min(am) + 1e-12
        )
        am = np.uint8(np.floor(am))
        m=np.argmax(am)
        m_min=np.argmin(am)
        r, c = divmod(m, am.shape[1])
        rmin, cmin = divmod(m_min, am.shape[1])
        erase_x.append(r)
        erase_y.append(c)
        erase_x_min.append(rmin)
        erase_y_min.append(cmin)

    erase_x=torch.tensor(erase_x).cuda()
    erase_y=torch.tensor(erase_y).cuda()
    erase_x_min=torch.tensor(erase_x_min).cuda()
    erase_y_min=torch.tensor(erase_y_min).cuda()
    sl = 0.02
    sh = 0.07
    r1 = 0.8
    img=images.clone()
    img_min=images.clone()
    for i in range(img.size(0)):
        for attempt in range(1000000000):
            area = img.size()[2] * img.size()[3]
            target_area_ratio =  random.uniform(sl, sh) + 0.03 * (1-gmmweight[i])
            target_area = target_area_ratio*area
            maxindex = torch.argmax(target_mask[i])
            flagmask = torch.zeros(target_mask[i].shape)
            flagmask[maxindex] = 1
            flagmask = flagmask.bool()
            target_mask[i][flagmask] -= target_area_ratio
            target_mask[i][~flagmask] += target_area_ratio/(target_mask[i].shape[-1]-1)
            aspect_ratio = random.uniform(r1, 1 / r1)
            h = int(round(math.sqrt(target_area*0.5 * aspect_ratio)))
            w = int(round(math.sqrt(target_area*0.5 / aspect_ratio)))
            if w < img.size()[3] and h < img.size()[2]:
                x1 = erase_x[i]
                y1 = erase_y[i]
                x1_min = erase_x_min[i]
                y1_min = erase_y_min[i]
                if x1+h>img.size()[2]:
                    x1=img.size()[2]-h
                if y1+w>img.size()[3]:
                    y1=img.size()[3]-w
                if x1_min+h>img.size()[2]:
                    x1_min=img.size()[2]-h
                if y1_min+w>img.size()[3]:
                    y1_min=img.size()[3]-w

                if img.size()[1] == 3:
                    img[i, 0, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                    img[i, 1, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)
                    img[i, 2, x1:x1 + h, y1:y1 + w] = random.uniform(0, 1)

                    img_min[i, 0, x1_min:x1_min + h, y1_min:y1_min + w] = random.uniform(0, 1)
                    img_min[i, 1, x1_min:x1_min + h, y1_min:y1_min + w] = random.uniform(0, 1)
                    img_min[i, 2, x1_min:x1_min + h, y1_min:y1_min + w] = random.uniform(0, 1)
                    break
    return img, img_min, target_mask

class MILLET(nn.Module):
    def __init__(self, feature_extractor, pool=None, args=None, pe='zeros', learn_pe=True, patch_dim=32, dropout=0.,
                 decoder=None, low_dim=None):
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
            average_stride = self.stride // self.patch_len  # self.shapelet_len // self.patch_len * 0.5 表示实际相对于原序列步幅为self.shapelet_len *0.5

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
        x = self.W_P(x.transpose(1,2).contiguous().view(B, patch_num, -1))  # x: [bs x nvars x patch_num x d_model]
        
        if self.args.backbone == 'MLP':
            x = self.act(x)





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


