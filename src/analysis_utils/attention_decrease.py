import torch
import os
import numpy as np

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.MultiTaskClassification import NonLinClassifier
from src.models.model import CNNAE

from src.models.MILLET_SALA import MILLET as NoisyPatchModel
from src.models.MILLET_only_patch import MILLET
from src.models.MILLET_model import (
    MILConjunctivePooling,
    GlobalAveragePooling,
    MILInstancePooling,
    MILAttentionPooling,
    MILAdditivePooling)
from src.models.FCN import FCNFeatureExtractor, FCNDecoder
from scipy.special import softmax

import os
import sys
import matplotlib
import math

sys.path.append(os.path.dirname(sys.path[0]))
import shutil
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyts import datasets
import torch.nn.functional as F

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score

# sys.path.append("..")

from src.utils.utils_ import select_class_by_class_with_GMM
from src.utils.saver import Saver
from src.utils.training_helper_single_model import main_wrapper_single_model
from src.utils.training_helper_global_sel_model import main_wrapper_global_sel_model
from src.ucr_data.load_ucr_pre import load_ucr
from src.uea_data.load_uea_pre import load_uea
from src.SALA_experiment_ucr import parse_args

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
columns = shutil.get_terminal_size().columns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################################################

def get_cmap(colour):
    return matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "white", colour])


def smooth_sequence(sequence, args):
    # window_len = int(args.len_shapelet[0] * len(sequence))
    window_len = int(max(0.05 * len(sequence), 5))
    window = np.ones(window_len) / window_len
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
    k_s = min(int(args._len_shapelet[0] * args.sample_len), args.patch_len)
    s = int(args.shapelet_stride)
    final_len = (math.floor(
        (args.sample_len - args.patch_len) / s) + 1) if args.patch else int(
        args.sample_len)
    p = max(int(((final_len - 1) * s + k_s - args.sample_len) / 2), 0)
    L = len(sequence)
    upscaled_sequence = np.full(L, -np.inf)

    num_windows = len(pool_sequence)

    for i in range(num_windows):
        start = i * s - p
        end = start + k_s

        upscaled_sequence[max(start, 0):min(end, L)] = np.maximum(upscaled_sequence[max(start, 0):min(end, L)],
                                                                  pool_sequence[i])

    return upscaled_sequence[p:p + L]


def plot_long_series_and_matched_shapelet_conf(x, interpretation, ground_truth, confident_id, Pred, args, fig_name):
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x_max = max(x.flatten())
    x_min = min(x.flatten())
    interpretation = softmax(interpretation, axis=1)
    interpretation_other_class = generate_interpretation_other_class(interpretation, Pred).squeeze()
    interpretations = interpretation[np.arange(len(x)), Pred].squeeze()
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

            # interpretation_x = interpretation[idx,pred].squeeze()
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
    plt.savefig(f'{args.dataset}_{fig_name}.png')
    plt.show()


def plot_long_series_and_matched_shapelet(x, interpretation, ground_truth, observed, Pred, args, fig_name, f1=''):
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x_max = max(x.flatten())
    x_min = min(x.flatten())
    # interpretation = softmax(interpretation, axis=2)
    itpmax = np.max(interpretation)
    itpmin = np.min(interpretation)
    interpretation = (interpretation - itpmin) / (itpmax - itpmin)
    interpretation_other_class = generate_interpretation_other_class(interpretation, Pred).squeeze()
    interpretations = interpretation[np.arange(len(x)), Pred].squeeze()
    # interpretations = interpretations - interpretation_other_class
    lim = max(np.abs(interpretations.flatten()))
    norm = plt.Normalize(-lim, lim)

    fig, axes = plt.subplots(args.nbins * 2, args.nbins, figsize=(12, 9), gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
    for c1 in range(args.nbins):
        for c2 in range(args.nbins):
            idxs = np.arange(len(ground_truth))[(ground_truth == c1) & (observed == c2)]
            if not idxs.size:
                axes[c1 * 2, c2].tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)
                axes[c1 * 2 + 1, c2].tick_params(axis='both', which='both', length=0, labelleft=False,
                                                 labelbottom=False)
                continue
            idx = idxs[0]
            pred = Pred[idx]
            x_np = x[idx].squeeze()
            interpretation_x = interpretations[idx].squeeze()
            if args.patch:
                interpretation_x = upscale_pooled_sequence(x_np, interpretation_x, args)

            ax = axes[c1 * 2, c2]
            ax.plot(x_np, color="black")
            plot_single_bg_heatmap(ax, interpretation_x, get_cmap(colours[2]), norm)
            ax.set_ylim(int(x_min * 1.1), int(x_max * 1.1))
            ax.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)

            ax.text(0.02, 0.98, f'Pred: {pred}', ha='left', va='top', transform=ax.transAxes, fontsize=8)

            ax_ = axes[c1 * 2 + 1, c2]
            if not args.patch:
                ax_.plot(smooth_sequence(interpretation_x, args), color=colours[2])
            else:
                ax_.plot(interpretation_x, color=colours[2])
            # ax_.set_ylim(-int(lim * 1.2), int(lim * 1.2))
            ax_.tick_params(axis='both', which='both', length=0, labelleft=True, labelbottom=False)  # 隐藏刻度线和刻度值

    fig.text(0.03, 0.5, 'ground truth', ha='center', va='center', rotation='vertical', fontsize=15)
    fig.text(0.5, 0.04, 'observed', ha='center', va='center', fontsize=15)
    fig.suptitle(args.dataset + fig_name, fontsize=17)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.93, hspace=0.2, wspace=0.2)
    plt.savefig(f'{args.dataset}_{fig_name}_f1{f1}.png')
    plt.show()



def plot_single_bg_heatmap(axis, values, cmap, norm, xs=None, alpha=0.8):
    if xs is None:
        xs = range(len(values))
    for x, s in zip(xs, values):
        colour = cmap(norm(s))
        axis.axvspan(x - 0.5, x + 0.5, color=colour, alpha=alpha, lw=0)


######################################################################################################
def clean_model(x_train, args):
    channel = x_train.shape[1] if args.patch_type != 'before_encode' else args.embedding_size
    # Network definition
    classifier1 = NonLinClassifier(args.embedding_size, args.nbins, d_hidd=args.classifier_dim, dropout=args.dropout,
                                   norm=args.normalization if not args.MILLET else 'layer')

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
            classifier=classifier1,
            args=args
        ),
        args=args,
        patch_dim=args.embedding_size,
        decoder=FCNDecoder(channel, args=args) if args.recon else None,
    ).to(device)
    model_to_save_dir = os.path.join(args.basicpath, 'model_save', args.dataset)

    model.load_state_dict(
        torch.load(os.path.join(model_to_save_dir, 'vanillaNoAug_sym0_20241007.pt'),
                   map_location=torch.device(f"cuda:{args.cuda_device}")))
    return model


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
    ytrue = np.concatenate(ytrue, axis=0)
    f1_weighted = f1_score(ytrue, preds.detach().numpy(), average='weighted')
    interpretation = torch.softmax(interpretation, dim=1)
    return interpretation, preds, confident_id, f1_weighted


def interpre_diff(interpre1, interpre2, labels):
    a = interpre1[np.arange(len(labels)), labels].squeeze()
    b = interpre2[np.arange(len(labels)), labels].squeeze()
    return nn.MSELoss()(a, b).item()


def main(args, dataset_name=None):
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

    model_path_dir = './src/model_save/'

    x_train = np.load(os.path.join(model_path_dir, dataset, 'sym30_x_train.npy'))
    Y_train = np.load(os.path.join(model_path_dir, dataset, 'sym30_y_train.npy'))
    Y_train_clean = np.load(os.path.join(model_path_dir, dataset, 'sym30_y_train_clean.npy'))
    x_valid = np.load(os.path.join(model_path_dir, dataset, 'sym30_x_valid.npy'))
    Y_valid = np.load(os.path.join(model_path_dir, dataset, 'sym30_y_valid.npy'))
    x_test = np.load(os.path.join(model_path_dir, dataset, 'sym30_x_test.npy'))
    Y_test = np.load(os.path.join(model_path_dir, dataset, 'sym30_y_test.npy'))

    classes = len(np.unique(Y_train))
    args.nbins = classes

    if args.MILLET:
        x_train = x_train.transpose(0, 2, 1)

    args.sample_len = x_train.shape[1]
    args.nvars = x_train.shape[2]

    args.patch_len = max(min(int(args.sample_len * args._len_shapelet[0]), args._patch_len),
                         1)  # MILLET_model.py , FCNDecoder 和 shapelet_utils.py中stride修改

    args.shapelet_stride = max(int(args.patch_len * args._shapelet_stride), 1)

    args.patch_stride = args.shapelet_stride

    args.num_classes = classes
    args.num_training_samples = len(x_train)
    args.k_val = min(np.median(np.bincount(Y_train_clean)).astype(int), args.k_val)


    batch_size = min(x_train.shape[0] // 10, args._batch_size)
    if x_train.shape[0] % batch_size == 1:
        batch_size += -1
    print(f'Batch size: {batch_size}')
    args.batch_size = batch_size
    args.test_batch_size = batch_size

    if args.model not in ['vanilla']:

        channel = x_train.shape[1] if args.patch_type != 'before_encode' else args.embedding_size
        # Network definition
        classifier = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                      norm=args.normalization if not args.MILLET else 'layer')
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
    else:  # 'vanilla'
        channel = x_train.shape[1] if args.patch_type != 'before_encode' else args.embedding_size
        # Network definition
        classifier1 = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim, dropout=args.dropout,
                                       norm=args.normalization if not args.MILLET else 'layer')

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
                classifier=classifier1,
                args=args
            ),
            args=args,
            patch_dim=args.embedding_size,
            decoder=FCNDecoder(channel, args=args) if args.recon else None,
        ).to(device)
    model_to_save_dir = os.path.join(args.basicpath, 'model_save', args.dataset)

    observed = Y_train

    # model_name = '{}.pt'.format(args.info)
    x_train = x_train.transpose(0, 2, 1)

    clean_vanilla = clean_model(x_train, args)
    interpretation_, pred_, confident_id_, f1_ = get_interpretation(clean_vanilla, x_train, Y_train_clean, observed,
                                                                    args)

    for model_name in ['{}'.format(args.outfile)]:  # '{}best'.format(args.outfile)]: #
        model.load_state_dict(
            torch.load(os.path.join(model_to_save_dir, model_name + '.pt'),
                       map_location=torch.device(f"cuda:{args.cuda_device}")))
        model.eval()
        model.to(device)
        interpretation, pred, confident_id, f1 = get_interpretation(model, x_train, Y_train_clean, observed, args)
        # diff = interpre_diff(interpretation, interpretation_, Y_train_clean)

        # plot_long_series_and_matched_shapelet(x_train, interpretation.numpy(), Y_train_clean, confident_id, pred.numpy(), args, model_name.split('.')[0])
        plot_long_series_and_matched_shapelet(x_train, interpretation.numpy(), Y_train_clean, observed, pred.numpy(), args, model_name.split('.')[0]+'_pred.pt',str(f1))
        plot_long_series_and_matched_shapelet(x_train, interpretation.numpy(), Y_train_clean, observed, observed, args, model_name.split('.')[0]+'_obs.pt',str(f1))
    return {'dataset': args.dataset, 'res': diff}


#####################################################################################################
if __name__ == '__main__':
#     comstr1 = '''--model vanilla --epochs 200 --warmup 20 --lr 1e-3 \
# --label_noise 0 --outfile vanillaNoAug_sym30_20241007 --group test --ni 0.3 \
# --embedding_size 128 --num_workers 0 --cuda_device 0 \
# --MILLET --pool Conjunctive --pseudo_loss_coef 1. --amp_noise 0.1 \
# --interpre_type interpretation --_shapelet_stride 0.5 --_patch_len 8 --_len_shapelet 0.05 \
# --p_threshold 0.6 --correct_threshold 0.7 --patch \
# --project test --patch_type before_encode --ucr 1 --recon \
# --L_rec_coef 0.1 --save_model --valid_set \
# --gamma 0.9 --group vanilla_sym30_1 --debug'''
#     comstr2 = '''--model global_sel \
# --epochs 200 --warmup 20 --lr 1e-3 --save_model \
# --label_noise 0 --outfile wo_aug_37_20241007_ --group test --ni 0.3 \
# --embedding_size 128 --num_workers 0 --cuda_device 1 --valid_set --select_type G_CbC_GMM \
# --MILLET --pool Conjunctive --pseudo_loss_coef 1. --amp_noise 0.1 \
# --interpre_type interpretation --_shapelet_stride 0.5 --_patch_len 8 --_len_shapelet 0.05 \
# --p_threshold 0.6 --correct_threshold 0.7 --patch --forward_type None --amp_mask 0.1 \
# --project test --patch_type before_encode --ucr 1 --label_correct_type soft --recon \
# --L_rec_coef 0.1 --consistency_loss_coef 0.1 \
# --only_max_min_noise max --only_max_min_mask max'''
    comstr3 = '''--model global_sel \
--epochs 200 --warmup 20 --lr 1e-3 --save_model \
--label_noise 0 --outfile pm_max_instance_37_20241007_best --group test --ni 0.3 \
--embedding_size 128 --num_workers 0 --cuda_device 1 --valid_set --select_type G_CbC_GMM \
--MILLET --pool Conjunctive --pseudo_loss_coef 1. --amp_noise 0.1 \
--interpre_type interpretation --_shapelet_stride 0.5 --_patch_len 8 --_len_shapelet 0.05 \
--p_threshold 0.6 --correct_threshold 0.7 --patch --forward_type None --amp_mask 0.1 \
--project test --patch_type before_encode --ucr 1 --label_correct_type soft --recon \
--L_rec_coef 0.1 --consistency_loss_coef 0.1 \
--only_max_min_noise max --only_max_min_mask max'''
    args = parse_args(comstr1)

    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    basicpath = os.path.dirname(father_path)

    print(f"father_path = {father_path}")
    result_value = []

    if args.ucr == 128:
        ucr = datasets.ucr_dataset_list()[args.from_ucr:args.end_ucr]
    else:
        ucr = ['ArrowHead', 'CBF', 'FaceFour', 'MelbournePedestrian', 'OSULeaf', 'Plane', 'Symbols', 'Trace']

    for i, comstr in enumerate([comstr1, comstr2, comstr3]):
        results = []

        for dataset in ucr:
            args = parse_args(comstr)
            args.dataset = dataset
            args.basicpath = basicpath

            results.append(main(args, args.dataset))

        pd.DataFrame(results).to_csv(f'{args.model}_{i}_sym30_interpre_diff_with_clean.csv')
