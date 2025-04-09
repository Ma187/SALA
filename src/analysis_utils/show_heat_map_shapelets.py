
import argparse
import os
import sys
import matplotlib
import math
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
print("os.getcwd: ",os.getcwd() )
sys.path.append('./')
from src.models.MultiTaskClassification import NonLinClassifier, MetaModel_AE, MetaModel
from src.models.model import CNNAE

from src.models.MILLET_SALA import MILLET as NoisyPatchModel
from src.models.MILLET_model import (
    MILLET, MILConjunctivePooling,
    GlobalAveragePooling,
    MILInstancePooling,
    MILAttentionPooling,
    MILAdditivePooling)
from src.models.FCN import FCNFeatureExtractor, FCNDecoder
from scipy.special import softmax


sys.path.append(os.path.dirname(sys.path[0]))
import shutil
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyts import datasets
import torch.nn.functional as F

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

# sys.path.append("..")

from src.utils.utils_ import create_synthetic_dataset,select_class_by_class_with_GMM
from src.utils.global_var import OUTPATH
from src.utils.saver import Saver
from src.utils.utils_scale import build_dataset_pt

from src.utils.training_helper_global_sel_model import main_wrapper_global_sel_model
from src.ucr_data.load_ucr_pre import load_ucr
from src.uea_data.load_uea_pre import load_uea

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
columns = shutil.get_terminal_size().columns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
######################################################################################################

def parse_args():
    # TODO: make external configuration file -json or similar.
    """
    Parse arguments
    """

    # Add global parameters
    parser = argparse.ArgumentParser(description='coteaching single experiment')

    # Synth Data
    parser.add_argument('--dataset', type=str, default='', help='UCR datasets')
    parser.add_argument('--outfile', type=str, default='CTW.csv', help='name of output file')
    parser.add_argument('--ni', type=float, default=0.5, help='label noise ratio')
    parser.add_argument('--label_noise', type=int, default=0, help='Label noise type, sym or int for asymmetric, '
                                                                   'number as str for time-dependent noise')

    parser.add_argument('--M', type=int, nargs='+', default=[20, 40, 60, 80])
    parser.add_argument('--reg_term', type=float, default=1,
                        help="Parameter of the regularization term, default: 0.")
    parser.add_argument('--alpha', type=float, default=32,
                        help='alpha parameter for the mixup distribution, default: 32')

    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_gradual', type=int, default=100)

    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--l2penalty', type=float, default=1e-4)

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed - only affects Network init')

    parser.add_argument('--classifier_dim', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=32)

    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--filters', nargs='+', type=int, default=[128, 128, 256, 256])
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--padding', type=int, default=2)

    # Suppress terminal out
    parser.add_argument('--disable_print', action='store_true', default=False)
    parser.add_argument('--plt_embedding', action='store_true', default=False)
    parser.add_argument('--plt_loss_hist', action='store_true', default=False)
    parser.add_argument('--plt_cm', action='store_true', default=False)
    parser.add_argument('--headless', action='store_true', default=False, help='Matplotlib backend')
    parser.add_argument('--beta',type=float,nargs='+',default=[0.,3.],help='the coefficient of model_loss2')
    parser.add_argument('--warmup',type=int,default=10,help='warmup epochs' )

    parser.add_argument('--model', choices=['co_teaching', 'co_teaching_mloss',
                                            'sigua', 'single_ae_aug_after_sel', 'single_aug', 'single_sel', 'vanilla',
                                            'single_aug_after_sel', 'single_ae_sel', 'single_ae', 'single_ae_aug',
                                            'single_ae_aug_sel_allaug', 'single_ae_aug_before_sel', 'dividemix', 'CTW',
                                            'CCR', 'ELR', 'shapelet_trans', 'single_shapelet_trans',
                                            'SALA',
                                            'global_sel_shapelet_trans', 'global_sel_conf_shapelet_trans'],
                        ## CCR: Class-Dependent-Label-Noise-Learning-with-Cycle-Consistency-Regularization
                        default='CTW')
    parser.add_argument('--manual_seeds', type=int, nargs='+', default=[37, 118, 337, 815, 19], # For fair comparation, we set the same seeds for all methods.
                        help='manual_seeds for five folds cross varidation')
    parser.add_argument('--label_correct_type', type=str, default='None', choices=['None','hard','soft'],
                        help='if correct label')
    parser.add_argument('--num_training_samples',type=int,default=0,help='num of trainging samples')
    # parser.add_argument('--loss', type=str, default='cores', help='type of loss function')
    parser.add_argument('--mixup', action='store_true', default=False, help='manifold mixup if or not')
    parser.add_argument('--mean_loss_len', type=int,default=1,help='the length of mean loss')
    parser.add_argument('--gamma', type=float, default=0.5, help='the weight of current sample loss in mean_loss_sel method')
    parser.add_argument('--arg_interval', type=int, default=1,
                        help='the batch-interval for augmentation in batch')
    parser.add_argument('--cuda_device', type=int, default=0, help='choose the cuda devcie')
    parser.add_argument('--aug', choices=['GNoise','NoAug','Oversample','Convolve','Crop','Drift','TimeWarp','Mixup'], default='NoAug')
    parser.add_argument('--sample_len', type=int,default=0)
    parser.add_argument('--ucr', type=int, default=0,help='if 128, run all ucr datasets')
    parser.add_argument('--basicpath', type=str, default='', help='basic path')
    parser.add_argument('--plot_tsne', action='store_true', default=False, help='if plot t-sne or not')
    parser.add_argument('--nbins', type=int, default=0, help='number of class')
    parser.add_argument('--save_model', action='store_true', default=False, help='if save model or not')
    parser.add_argument('--from_ucr', type=int, default=0, help='begin from which dataset')
    parser.add_argument('--end_ucr', type=int, default=128, help='end at which dataset')
    parser.add_argument('--sel_method', type=int, default=3,choices=[0,1,2,3,4],
                        help='''0: select ratio is known (co-teaching, sigua);
                                1,2: select confident samples class by class;
                                3: select w/ EPS
                                4: select w/o EPS''')
    parser.add_argument('--tsne_during_train', action='store_true', default=False, help='if plot tsne during training or not')
    parser.add_argument('--tsne_epochs', type=int, nargs='+', default=[49, 99, 149, 199, 249,299],
                        help='manual_seeds for five folds cross varidation')

    parser.add_argument('--augMSE', action='store_true', default=False, help='if use MSE on aug or not')
    parser.add_argument('--bad_weight', type=float, default=1e-3,help='for sigua')
    parser.add_argument('--aug_ae', action='store_true', default=False, help='if reconstruct augmented samples or not')
    parser.add_argument('--window', type=str, choices=['single', 'all'], default='all',
                        help='single_train/single_test: only plot training/test data; all: plot all data ')
    parser.add_argument('--L_aug_coef', type=float, default=1.,
                        help='the coefficient of L_aug')
    parser.add_argument('--L_rec_coef', type=float, default=1.,
                        help='the coefficient of L_rec')
    parser.add_argument('--confcsv', type=str, default=None,
                        help='the file of saving conf_num')
    parser.add_argument('--whole_data_select', action='store_true', default=False,
                        help='if select from whole data')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--plt_loss_density', action='store_true', default=False,
                        help='if plot loss density')
    parser.add_argument('--standardization_choice', type=str, choices=['z-score', 'min-max'], default='z-score',
                        help='choose the method of standardization')
    parser.add_argument('--debug', action='store_true', default=False,help='')
    parser.add_argument('--valid_set', action='store_true', default=False,help='')

    parser.add_argument('--sess', default='default', type=str, help='session id')
    parser.add_argument('--start_prune', default=40, type=int,
                        help='')

    ## CCR
    parser.add_argument('--lam', type=float, default=0.3)
    parser.add_argument('--init', type=float, default=5)
    parser.add_argument('--anchor', action='store_false')

    parser.add_argument('--_len_shapelet', type=float, nargs='+', default=[0.2], help="predefined args")
    parser.add_argument('--len_shapelet', type=int, nargs='+', default=[0.2], help="子序列/patch的长度比例")
    parser.add_argument('--nb_shapelet_cls', type=int, default=25)
    parser.add_argument('--nb_shapelet', type=int, default=25)
    parser.add_argument('--sample_spec_shapelet', action='store_true', default=False)
    parser.add_argument('--general_shapelet', action='store_true', default=False)
    parser.add_argument('--class_shapelet', action='store_true', default=False)
    parser.add_argument('--lambda_shapelet', type=float, default=0.1)
    parser.add_argument('--soft_alpha', type=float, default=-1)
    parser.add_argument('--l1_reg', type=float, default=0.)
    parser.add_argument('--alpha_aux_loss', type=float, default=0.01)
    parser.add_argument('--shapelet_fusion', type=str, default='none', choices=['none', 'inst_fusion', 'emb_fusion'])
    parser.add_argument('--select_type', type=str, default='G_GMM', choices=['G_GMM', 'G_CbC_GMM'])
    parser.add_argument('--just_warmup', action='store_true', default=False)
    parser.add_argument('--MILLET', action='store_true', default=False, help='base on MILLET')
    parser.add_argument('--pool', type=str, default='Conjunctive', help='base on MILLET', choices=[
        'Conjunctive',
        'GlobalAverage',
        'Instance',
        'Attention',
        'Additive'
    ])

    parser.add_argument('--_shapelet_stride', type=float, default=0.5)
    parser.add_argument('--shapelet_stride', type=float, default=0.5)
    parser.add_argument('--interpre_type', type=str, default='interpretation',
                        choices=['interpretation', 'atten'], help='')
    ## wandb
    parser.add_argument('--group', type=str, default='my group')
    parser.add_argument('--exp_name', type=str, default='my exp')

    parser.add_argument('--patch', action='store_true', default=False)
    parser.add_argument('--info', type=str, default='')

    parser.add_argument('--patch_len', type=int, default=8, help='')
    parser.add_argument('--_patch_len', type=int, default=8, help='')
    parser.add_argument('--patch_type', type=str, default='after_encode', choices=['before_encode', 'after_encode'])

    parser.add_argument('--forward_type', default='None',type=str,choices=['None','perturb','mask','pm','mm','pp','mp'])
    parser.add_argument('--only_max_min', default='None',type=str,choices=['None','max_min','max'])
    parser.add_argument('--amp_noise', type=float, default=0.)
    parser.add_argument('--amp_mask', type=float, default=0.)
    parser.add_argument('--mean_norm', type=int, default=0)
    parser.add_argument('--consistency_loss_coef', type=float, default=0.)
    parser.add_argument('--pseudo_loss_coef', type=float, default=0.)
    parser.add_argument('--recon', action='store_true', default=False)
    parser.add_argument('--correct_threshold', type=float, default=0.7)
    parser.add_argument('--project', type=str, default='Shapelets')
    parser.add_argument('--nvars', type=int, default=1)
    parser.add_argument('--backbone', type=str, default="FCN", choices=['FCN', 'MLP'])

    # Add parameters for each particular network

    # args = parser.parse_args()
    args = parser.parse_args("--model global_sel --epochs 800 --warmup 800 --just_warmup --lr 1e-3 \
--label_noise 0 --outfile vanilla_synthetic_sym30_800 --group test --ni 0.3 --save_model \
--embedding_size 128 --num_workers 0 --cuda_device 1 --valid_set --select_type G_CbC_GMM \
--MILLET --pool Conjunctive --pseudo_loss_coef 1. --amp_noise 0.1 --manual_seeds 37 \
--interpre_type interpretation --_shapelet_stride 0.5 --_patch_len 8 --_len_shapelet 0.05 \
--p_threshold 0.6 --correct_threshold 0.7 --patch --forward_type pm --amp_mask 0.1 \
--project test --patch_type before_encode --ucr 128 --label_correct_type soft --recon \
--L_rec_coef 0.1 --consistency_loss_coef 0.1 --from_ucr 0 --dataset synthetic_3Class \
--info vanilla_synthetic3C_sym30_37_20250120_".split(' '))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device==torch.device('cuda'):
        torch.cuda.set_device(args.cuda_device)

    return args

def get_cmap(colour):
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "white", colour])


def smooth_sequence(sequence, args):
    # window_len = int(args.len_shapelet[0] * len(sequence))
    window_len = int(max(0.05 * len(sequence),5))
    # 构建窗口
    window = np.ones(window_len) / window_len
    # 使用convolve函数进行移动平均
    smoothed_seq = np.convolve(sequence, window, mode='same')

    return smoothed_seq

def generate_interpretation_other_class(interpretation, pred):
    batch_size, num_classes, seq_length = interpretation.shape
    interpretation_other_class = np.zeros((batch_size, seq_length))

    for i in range(batch_size):
        pred_label = int(pred[i])
        other_class_indices = np.setdiff1d(np.arange(num_classes), pred_label, assume_unique=True)
        other_class_weights = interpretation[i, other_class_indices, :].mean(axis=0)
        interpretation_other_class[i, :] = other_class_weights

    return interpretation_other_class

def upscale_pooled_sequence(sequence, pool_sequence, args):
    k_s = min(int(args._len_shapelet[0]*args.sample_len), args.patch_len)
    s = int(args.shapelet_stride)
    final_len = (math.floor(
        (args.sample_len - args.patch_len) / s) + 1) if args.patch else int(
        args.sample_len)
    p = max(int(((final_len-1)*s+k_s-args.sample_len)/2), 0)
    L = len(sequence)
    upscaled_sequence = np.full(L, -np.inf)

    num_windows = len(pool_sequence)

    for i in range(num_windows):
        start = i * s - p
        end = start + k_s

        upscaled_sequence[max(start, 0):min(end, L)] = np.maximum(upscaled_sequence[max(start, 0):min(end, L)], pool_sequence[i])

    return upscaled_sequence[p:p+L]

def plot_long_series_and_matched_shapelet_conf(x, interpretation, ground_truth, confident_id, Pred, args, fig_name):
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x_max = max(x.flatten())
    x_min = min(x.flatten())
    interpretation = softmax(interpretation, axis=1)
    interpretation_other_class = generate_interpretation_other_class(interpretation,Pred).squeeze()
    interpretations = interpretation[np.arange(len(x)),Pred].squeeze()
    interpretations = interpretations - interpretation_other_class
    lim = max(np.abs(interpretations.flatten()))
    norm = plt.Normalize(-lim, lim)
    max_columns = 4
    fig, axes = plt.subplots(args.nbins * 2, max_columns, figsize=(12, 9), gridspec_kw={'hspace': 0.1, 'wspace': 0.1})

    for ax_row in axes:
        for ax in ax_row:
            ax.set_visible(False)

    for c1 in range(args.nbins):
        class_idxs = [idx for idx in confident_id if ground_truth[idx] == c1]
        for idx_count, idx in enumerate(class_idxs[:min(max_columns, len(class_idxs))]):
            pred = Pred[idx]
            x_np = x[idx].squeeze()

            interpretation_x = interpretations[idx].squeeze()
            if args.patch:
                interpretation_x = upscale_pooled_sequence(x_np, interpretation_x, args)

            ax = axes[c1 * 2, idx_count]
            ax.set_visible(True)
            ax.plot(x_np, color="black")
            plot_single_bg_heatmap(ax, interpretation_x, get_cmap(colours[2]), norm)
            ax.set_ylim(int(x_min * 1.1), int(x_max * 1.1))
            ax.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)
            ax.text(0.02, 0.98, f'Pred: {pred}', ha='left', va='top', transform=ax.transAxes, fontsize=8)

            ax_ = axes[c1 * 2 + 1, idx_count]
            ax_.set_visible(True)
            if not args.patch:
                ax_.plot(smooth_sequence(interpretation_x, args), color=colours[2])
            else:
                ax_.plot(interpretation_x, color=colours[2])
            ax_.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)

    fig.text(0.03, 0.5, 'ground truth', ha='center', va='center', rotation='vertical', fontsize=15)
    fig.text(0.5, 0.04, 'observed', ha='center', va='center', fontsize=15)
    fig.suptitle(args.dataset + (' Best Model' if 'best' in fig_name else ' Last Model'), fontsize=17)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.93, hspace=0.2, wspace=0.2)
    plt.savefig(f'{args.dataset}_{fig_name}.pdf')
    plt.show()


def plot_long_series_and_matched_shapelet(x, interpretation, ground_truth, observed, Interpre_label, args, fig_name,f1='',interpre_label_type='Pred'):
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x_max = max(x.flatten())
    x_min = min(x.flatten())
    # interpretation = softmax(interpretation, axis=2)
    itpmax = np.max(interpretation)
    itpmin = np.min(interpretation)
    interpretation = (interpretation - itpmin) / (itpmax-itpmin)
    # interpretation_other_class = generate_interpretation_other_class(interpretation,Interpre_label).squeeze()
    interpretations = interpretation[np.arange(len(x)),Interpre_label].squeeze()#-interpretation[np.arange(len(x)),Interpre_label].squeeze()
    # interpretations = interpretations - interpretation_other_class
    lim = max(np.abs(interpretations.flatten()))
    norm = plt.Normalize(-lim, lim)

    # fig, axes = plt.subplots(args.nbins * 2, args.nbins, figsize=(12, 9), gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
    for c1 in range(args.nbins):
        for c2 in range(args.nbins):

            idxs = np.arange(len(ground_truth))[(ground_truth == c1) & (observed == c2)]
            if not idxs.size:
                # axes[c1 * 2, c2].tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)
                # axes[c1 * 2 + 1, c2].tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)
                continue

            idx = idxs[0]

            for k in range(len(idxs)):

                fig, axes = plt.subplots(1, 1, figsize=(4, 1.5),
                                         gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
                axes.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)
                idx = idxs[k]
                interpre_label = Interpre_label[idx]
                x_np = x[idx].squeeze()
                interpretation_x = interpretations[idx].squeeze()
                if args.patch:
                    interpretation_x = upscale_pooled_sequence(x_np, interpretation_x, args)

                ax = axes
                ax.plot(x_np, color="black")
                plot_single_bg_heatmap(ax, interpretation_x, get_cmap(colours[2]), norm)
                ax.set_ylim(int(x_min * 1.1), int(x_max * 1.1))
                ax.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)

                ax.text(0.02, 0.98, f'{interpre_label_type}:{interpre_label}', ha='left', va='top', transform=ax.transAxes, fontsize=17)

                # ax_ = axes[c1 * 2 + 1, c2]
                # if not args.patch:
                #     ax_.plot(smooth_sequence(interpretation_x, args), color=colours[2])
                # else:
                #     ax_.plot(interpretation_x, color=colours[2])
                # ax_.set_ylim(-int(lim * 1.2), int(lim * 1.2))
                # ax_.tick_params(axis='both', which='both', length=0, labelleft=True, labelbottom=False)  # 隐藏刻度线和刻度值

                # fig.text(0.03, 0.5, 'ground truth', ha='center', va='center', rotation='vertical', fontsize=15)
                # fig.text(0.5, 0.04, 'observed', ha='center', va='center', fontsize=15)
                # fig.suptitle(args.dataset + (' Best Model' if 'best' in fig_name else ' Last Model'),fontsize=12)

                plt.tight_layout()
                plt.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.93, hspace=0.2, wspace=0.2)
                tmp = ('best' if 'best' in fig_name else 'last')

                plt.savefig(f'./src/analysis_utils/单图/{args.dataset}_{tmp}_{fig_name.split(".")[0]}_gt{c1}_obs{c2}_interpre{interpre_label}_id{idx}.png')
                # plt.show()

# def plot_long_series_and_matched_shapelet(x, interpretation, ground_truth, observed,Pred, args,fig_name):
#     colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#     lim = max(np.abs(interpretation.flatten()))
#     x_max = max(x.flatten())
#     x_min = min(x.flatten())
#     norm = plt.Normalize(-lim, lim)
#
#     fig, axes = plt.subplots(args.nbins * 2, args.nbins,figsize=(12,9))
#     for c1 in range(args.nbins):
#         for c2 in range(args.nbins):
#             try:
#                 if c1 == c2:
#                     idx = np.arange(len(ground_truth))[(ground_truth == c1) & (Pred == c1)][0]
#                 else:
#                     idx = np.arange(len(ground_truth))[(ground_truth==c1) & (observed==c2)][0]
#                 pred = Pred[idx]
#                 x_np = x[idx].squeeze()
#                 interpretation_x = interpretation[idx].squeeze()[c2]
#                 if args.patch:
#                     interpretation_x = upscale_pooled_sequence(x_np, interpretation_x, args)
#
#                 ax = axes[c1*2, c2]
#                 ax.plot(x_np, color="black")
#                 plot_single_bg_heatmap(ax, interpretation_x, get_cmap(colours[2]), norm)
#                 ax.set_ylim(int(x_min*1.1), int(x_max*1.1))
#                 ax.set_title('{}'.format(pred))
#                 ax.set_xticklabels([])
#                 ax.set_yticklabels([])
#
#                 ax_ = axes[c1*2 + 1, c2]
#                 ax_.plot(interpretation_x, color=colours[2])
#                 ax_.set_ylim(-int(lim * 1.2), int(lim * 1.2))
#                 ax_.set_xticklabels([])
#                 ax_.set_yticklabels([])
#
#             except:
#                 continue
#
#     # plt.xlabel("time stamp",fontdict={'fontsize':20})
#     fig.text(0.03, 0.47, 'ground truth',ha='center', rotation='vertical',fontsize=15)
#     fig.text(0.491, 0.02, 'observed',ha='center',fontsize=15)
#     fig.suptitle(args.dataset + (' Best Model' if 'best' in fig_name else ' Last Model'),fontsize=17)
#     plt.tight_layout()
#     plt.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.93)
#     plt.savefig(f'{args.dataset}_{fig_name}.png')
#     plt.show()

def plot_single_bg_heatmap(axis, values, cmap, norm, xs=None, alpha=0.8):
    if xs is None:
        xs = range(len(values))
    for x, s in zip(xs, values):
        colour = cmap(norm(s))
        axis.axvspan(x - 0.5, x + 0.5, color=colour, alpha=alpha, lw=0)

######################################################################################################

def get_interpretation(model, x, y_clean, observed, args):
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(observed).long(),
                                  torch.from_numpy(np.arange(len(observed))), torch.from_numpy(y_clean))
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers)

    stride = int(args.shapelet_stride)

    final_len = (math.floor(
        (args.sample_len - args.patch_len) / stride) + 1) if args.patch else int(
        args.sample_len)
    interpretation = torch.zeros(args.num_training_samples, args.nbins, final_len)
    preds = torch.zeros(args.num_training_samples).long()
    loss_all = np.zeros(args.num_training_samples)
    ytrue = []
    with torch.no_grad():
        for batch_idx, (x, y_hat, x_idx, y_true) in enumerate(data_loader):
            x = x.to(device)
            y_hat = y_hat.to(device)
            ytrue.append(y_true.numpy())
            if args.MILLET:
                out_dict = model(x)
                interpretation[x_idx] = out_dict['interpretation'].detach().cpu()
                preds[x_idx] = torch.argmax(out_dict['bag_logits'], dim=1).detach().cpu().long()
                model_loss = nn.CrossEntropyLoss(reduce=False)(out_dict['bag_logits'], y_hat)
                loss_all[x_idx] = model_loss.data.detach().clone().cpu().numpy()

    confident_id, conf_score = select_class_by_class_with_GMM(loss_all=loss_all, args=args,
                                                              labels=torch.tensor(observed),
                                                              p_threshold=args.p_threshold)
    ytrue = np.concatenate(ytrue,axis=0)
    f1_weighted = f1_score(ytrue, preds.detach().numpy(), average='weighted')

    return interpretation, preds, confident_id,f1_weighted


def main(args, dataset_name=None):

    saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0],
                  hierarchy=os.path.join(args.dataset),args=args)

    if args.plot_tsne:
        args.save_model=True

    ######################################################################################################
    print(f'{args}')

    ######################################################################################################
    SEED = args.seed
    # TODO: implement multi device and different GPU selection
    torch.manual_seed(SEED)
    if device == 'cuda':
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    if args.headless:
        print('Setting Headless support')
        plt.switch_backend('Agg')
    else:
        backend = 'Qt5Agg'
        print(f'Swtiching matplotlib backend to {backend}')
        # plt.switch_backend(backend)

    # X, Y = load_data(args.dataset)
    if 'synthetic' in args.dataset:
        # X, Y=create_synthetic_dataset(ts_n=800)

        model_path_dir = './src/model_save/'
        x_train = np.load(os.path.join(model_path_dir, args.dataset, 'sym30_x_train.npy'))
        Y_train = np.load(os.path.join(model_path_dir, args.dataset, 'sym30_y_train.npy'))
        x_test = np.load(os.path.join(model_path_dir, args.dataset, 'sym30_x_test.npy'))
        Y_test = np.load(os.path.join(model_path_dir, args.dataset, 'sym30_y_test.npy'))
        x_valid = np.load(os.path.join(model_path_dir, args.dataset, 'sym30_x_valid.npy'))
        Y_valid = np.load(os.path.join(model_path_dir, args.dataset, 'sym30_y_valid.npy'))
        Y_train_clean = np.load(os.path.join(model_path_dir, args.dataset, 'sym30_y_train_clean.npy'))

    # elif args.dataset in datasets.uea_dataset_list():
    #     X, Y = load_uea(args.dataset)
    # else:
    #     X, Y = load_ucr(args.dataset)
    #     # x_train, Y_train_clean, x_test, Y_test_clean = load_ucr(args.dataset)
    #
    # if args.MILLET:
    #     x_train = x_train.transpose(0, 2, 1)

    else:
        x_train, Y_train_clean, x_test, Y_test, classes, val_dataset, val_target = build_dataset_pt(args)

    args.sample_len = x_train.shape[2]
    args.patch_len = max(min(int(args.sample_len * args._len_shapelet[0]), args._patch_len),
                             1)  # MILLET_model.py , FCNDecoder 和 shapelet_utils.py中stride修改

    if args.patch:
       args.shapelet_stride = max(int(args.patch_len * args._shapelet_stride), 1)

    args.num_training_samples = len(x_train)
    classes = len(np.unique(Y_train_clean))
    args.nbins = classes

    # observed, _ = flip_label(x_train, Y_train_clean, args.ni, args)

    batch_size = min(x_train.shape[0] // 10, args.batch_size)
    args.nvars = x_train.shape[1]
    if x_train.shape[0] % batch_size == 1:
        batch_size += -1
    print(f'Batch size: {batch_size}')
    args.batch_size = batch_size
    args.test_batch_size = batch_size

    classes = len(np.unique(Y_train_clean))
    args.nbins = classes
    print(f'\nClasses: {classes}\n')
    channel = x_train.shape[1] if args.patch_type != 'before_encode' else args.embedding_size
    # Network definition
    classifier = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                  norm=args.normalization if not args.MILLET else 'layer')
    if args.MILLET and args.model=='vanilla':

        pooling_dict = {
            'Conjunctive': MILConjunctivePooling,
            'GlobalAverage': GlobalAveragePooling,
            'Instance': MILInstancePooling,
            'Attention': MILAttentionPooling,
            'Additive': MILAdditivePooling
        }
        model = MILLET(
            FCNFeatureExtractor(channel, seq_len=args.sample_len),
            pooling_dict[args.pool](
                args.embedding_size,
                args.nbins,
                dropout=0.1,
                apply_positional_encoding=True,
                classifier=classifier,
                args=args
            ),
            args=args,
            patch_dim=args.embedding_size,
            decoder=FCNDecoder(channel, args=args) if args.recon else None,
        ).to(device)
    elif args.MILLET:
        pooling_dict = {
            'Conjunctive': MILConjunctivePooling,
            'GlobalAverage': GlobalAveragePooling,
            'Instance': MILInstancePooling,
            'Attention': MILAttentionPooling,
            'Additive': MILAdditivePooling
        }

        backbone = NoisyPatchModel if args.patch else MILLET
        model = backbone(
            FCNFeatureExtractor(channel, seq_len=args.sample_len),
            pooling_dict[args.pool](
                args.embedding_size,
                args.nbins,
                dropout=0.1,
                apply_positional_encoding=True,
                classifier=classifier,
                args=args
            ),
            args=args,
            patch_dim=args.embedding_size,
            decoder=FCNDecoder(channel, args=args) if args.recon else None,
        ).to(device)
    else:
        model = CNNAE(input_size=x_train.shape[2], num_filters=args.filters, embedding_dim=args.embedding_size,
                      seq_len=x_train.shape[1], kernel_size=args.kernel_size, stride=args.stride,
                      padding=args.padding, dropout=args.dropout, normalization=args.normalization).to(device)

        ######################################################################################################
        # model is multi task - AE Branch and Classification branch
        # model = MetaModel_AE(ae=model, classifier=classifier, name='CNN').to(device)
        model = MetaModel(ae=model, classifier=classifier, name='CNN').to(device)

    model_to_save_dir = os.path.join(args.basicpath, 'model_save', args.dataset)

    observed = np.load(os.path.join(model_to_save_dir, '{}obs_label.npy'.format(args.info)))

    # model_name = '{}.pt'.format(args.info)
    for model_name in ['{}'.format(args.info), '{}best'.format(args.info)]:
        model.load_state_dict(
            torch.load(os.path.join(model_to_save_dir, model_name+'.pt'), map_location=torch.device(f"cuda:{args.cuda_device}")))
        model.eval()
        model.to(device)
        interpretation, pred, confident_id, f1 = get_interpretation(model, x_train, Y_train_clean, observed, args)
        # plot_long_series_and_matched_shapelet(x_train, interpretation.numpy(), Y_train_clean, confident_id, pred.numpy(), args, model_name.split('.')[0])
        plot_long_series_and_matched_shapelet(x_train, interpretation.numpy(), Y_train_clean, observed, pred.numpy(), args, model_name.split('.')[0]+'_pred',str(f1),interpre_label_type='Pred')
        plot_long_series_and_matched_shapelet(x_train, interpretation.numpy(), Y_train_clean, observed, observed, args, model_name.split('.')[0]+'_obs',str(f1),interpre_label_type='Obs')
######################################################################################################


if __name__ == '__main__':
    args = parse_args()

    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    basicpath = os.path.dirname(father_path)

    print(f"father_path = {father_path}")
    result_value = []

    if args.ucr==128:
        ucr=datasets.ucr_dataset_list()[args.from_ucr:args.end_ucr]
    else:
        ucr = ['ArrowHead','CBF','FaceFour','MelbournePedestrian','OSULeaf','Plane','Symbols','Trace']

    if args.dataset == '':
        for dataset in ucr:
            args = parse_args()
            args.dataset = dataset
            args.basicpath = basicpath

            main(args, args.dataset)
    else:
        args = parse_args()
        args.basicpath = basicpath

        main(args, args.dataset)

