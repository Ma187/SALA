import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

father_path = os.path.abspath(os.path.dirname(curPath) + os.path.sep + ".")
basicpath = os.path.dirname(father_path)

import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.utils_scale import downsample_torch, set_seed, adjust_learning_rate, build_loss, \
    construct_lp_graph, calculate_scale_flow_jsd, get_clean_class_jsd_ind, get_clean_loss_ind, get_clean_class_loss_ind, evaluate_scale_flow_acc


from src.utils.utils_ import flip_label
import shutil
import warnings
import pandas as pd
from pyts import datasets
from sklearn.model_selection import StratifiedKFold
from src.models.MultiTaskClassification import NonLinClassifier
from src.ucr_data.load_ucr_pre import load_ucr
from src.models.MILLET_only_patch import MILLET
from src.models.FCN import FCNFeatureExtractor, FCNDecoder
from src.models.MILLET_model import (
    MILConjunctivePooling,
    GlobalAveragePooling,
    MILInstancePooling,
    MILAttentionPooling,
    MILAdditivePooling)

columns = shutil.get_terminal_size().columns
warnings.filterwarnings("ignore")
def parse_args():

    parser = argparse.ArgumentParser()

    # Base setup
    parser.add_argument('--backbone', type=str, default='fcn', help='encoder backbone, fcn')
    parser.add_argument('--random_seed', type=int, default=42, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default='SleepEDF',
                        help='')  # ['SleepEDF', 'FD-A']
    parser.add_argument('--archive', type=str, default='UCR',
                        help='Four, UCR, UEA')
    parser.add_argument('--num_classes', type=int, default=0, help='number of class')
    parser.add_argument('--input_size', type=int, default=1, help='input_size')

    # Label noise
    parser.add_argument('--label_noise', type=int, default=0,
                        help='0 is Sym, 1 is Asym, -1 is Instance')
    parser.add_argument('--label_noise_rate', type=float, default=0.5,
                        help='label noise ratio, sym: 0.2, 0.5, asym: 0.4, ins: 0.4')
    parser.add_argument('--warmup_epoch', type=int, default=30, help='30 or 50')
    parser.add_argument('--small_loss_criterion', type=int, default=1, help='1 is use the warm_up small loss, 0 is use the jsd small loss.')
    parser.add_argument('--scale_nums', type=int, default=3, help='3, 4, 5, 6')
    parser.add_argument('--scale_list', type=list, default=[1, 2, 4], help='')
    parser.add_argument('--knn_num', type=int, default=10, help='')
    parser.add_argument('--moment_alpha', type=float, default=0.9, help='')
    parser.add_argument('--epoch_decay_start', type=int, default=80)
    parser.add_argument('--epoch_correct_start', type=int, default=120)

    # training setup
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--epoch', type=int, default=200, help='training epoch')
    parser.add_argument('--cuda_device', type=int, default=0, help='choose the cuda devcie')


    # classifier setup
    parser.add_argument('--classifier', type=str, default='linear', help='')
    parser.add_argument('--classifier_input', type=int, default=128, help='input dim of the classifiers')

    parser.add_argument('--patch_type', type=str, default='after_encode', choices=['before_encode', 'after_encode'])
    parser.add_argument('--recon', action='store_true', default=False)
    parser.add_argument('--ucr', type=int, default=0, help='if 128, run all ucr datasets')
    parser.add_argument('--from_ucr', type=int, default=0, help='begin from which dataset')
    parser.add_argument('--outfile', type=str, default='CTW.csv', help='name of output file')
    parser.add_argument('--patch', action='store_true', default=False)
    parser.add_argument('--embedding_size', type=int, default=32)
    parser.add_argument('--MILLET', action='store_true', default=False, help='base on MILLET')
    parser.add_argument('--pool', type=str, default='Conjunctive', help='base on MILLET', choices=[
        'Conjunctive',
        'GlobalAverage',
        'Instance',
        'Attention',
        'Additive'
    ])
    parser.add_argument('--interpre_type', type=str, default='interpretation',
                        choices=['interpretation', 'atten'], help='')
    parser.add_argument('--valid_set', action='store_true', default=False, help='')
    parser.add_argument('--_shapelet_stride', type=float, default=0.1, help="predefined args")
    parser.add_argument('--_patch_len', type=int, default=8, help='predefined args')
    parser.add_argument('--_len_shapelet', type=float, nargs='+', default=[0.2], help="predefined args")
    parser.add_argument('--_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--classifier_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--mean_norm', type=int, default=0)
    parser.add_argument('--model')
    parser.add_argument('--end_ucr', type=int, default=128, help='end at which dataset')
    parser.add_argument('--normalization', type=str, default='batch')



    # 脚本运行时使用以下代码
    args = parser.parse_args()
    # 调试时使用以下代码，注释以上代码
#     args = parser.parse_args("--model scale --epoch 200 --warmup_epoch 20 --patch_type before_encode \
# --recon --label_noise 0 --label_noise_rate 0.3 --knn_num 5 \
# --ucr 1 --from_ucr 0 --outfile scale_test3_epoch200_test.csv --cuda_device 0 --patch \
# --embedding_size 128 --MILLET --pool Conjunctive --interpre_type interpretation --valid_set --lr 1e-3 \
# --_shapelet_stride 0.5 --_patch_len 8 --_len_shapelet 0.05".split(' '))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device('cuda'):
        torch.cuda.set_device(args.cuda_device)
    set_seed(args)

    return args
def experiment(args):
    # define drop rate schedule
    rate_schedule = np.ones(args.epoch) * args.label_noise_rate
    rate_schedule[:args.warmup_epoch] = np.linspace(0, args.label_noise_rate, args.warmup_epoch)

    result_value = []

    if args.ucr == 128:
        ucr = datasets.ucr_dataset_list()[args.from_ucr:args.end_ucr]
        # ucr = datasets.ucr_dataset_list()[args.from_ucr:args.end_ucr][::-1]
    elif args.ucr == 1:
        ucr = ['ArrowHead', 'CBF','FaceFour', 'MelbournePedestrian',
               'OSULeaf', 'Plane', 'Symbols', 'Trace']
        # ucr = ['ArrowHead']

    else:
        ucr = ['ArrowHead', 'CBF', 'FaceFour', 'MelbournePedestrian',
               'OSULeaf', 'Plane', 'Symbols', 'Trace']


    for dataset_name in ucr:
        args = parse_args()
        args.dataset = dataset_name

        df_results = main(args, dataset_name, rate_schedule)
        result_value.append(df_results)

        if args.label_noise == -1:
            label_noise = 'inst'
        elif args.label_noise == 0:
            args.noise_type = 'symmetric'
            label_noise = 'sym'
        else:
            args.noise_type = 'asymmetric'
            label_noise = "asym"

        path = os.path.abspath(os.path.join(father_path, 'statistic_results', args.outfile))
        print(path)
        pd.DataFrame(result_value).to_csv(path)

def main(args, dataset_name=None, rate_schedule = None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('*' * shutil.get_terminal_size().columns)
    print('UCR Dataset: {}'.format(args.dataset).center(columns))
    print('*' * shutil.get_terminal_size().columns)


    result_evalution = dict()

    X, Y = load_ucr(args.dataset)

    if len(X) <= 400:
        args.lr = 1e-4

    classes = len(np.unique(Y))
    args.nbins = classes
    args.sample_len = X.shape[1]
    args.nvars = X.shape[2]

    args.patch_len = max(min(int(args.sample_len * args._len_shapelet[0]), args._patch_len),
                         1)  # MILLET_model.py , FCNDecoder 和 shapelet_utils.py中stride修改

    args.shapelet_stride = max(int(args.patch_len * args._shapelet_stride), 1)

    args.patch_stride = args.shapelet_stride
    # 避免nb_shapelet超出可选取的子序列数量

    skf = StratifiedKFold(n_splits=5)
    id_acc = 0
    starttime = time.time()

    if args.MILLET:
        X = X.transpose(0, 2, 1)





    for kk, (trn_index, test_index) in enumerate(skf.split(X, Y)):
        id_acc = id_acc + 1
        x_train = X[trn_index]
        x_test = X[test_index]
        Y_train_clean = Y[trn_index]
        Y_test_clean = Y[test_index]

        x_valid = None
        y_valid_clean = None
        if args.valid_set:
            train_valid_skf = StratifiedKFold(n_splits=4)
            trn_index, val_index = next(iter(train_valid_skf.split(x_train, Y_train_clean)))  # 只需要划分一次
            x_valid = x_train[val_index]
            y_valid_clean = Y_train_clean[val_index]
            x_train = x_train[trn_index]
            Y_train_clean = Y_train_clean[trn_index]

        print("id_acc = ", id_acc, Y_train_clean.shape, Y_test_clean.shape)

        args.num_classes = classes
        args.num_training_samples = len(x_train)

        batch_size = min(x_train.shape[0] // 10, args._batch_size)
        if x_train.shape[0] % batch_size == 1:
            batch_size += -1
        print(f'Batch size: {batch_size}')
        args.batch_size = batch_size
        args.test_batch_size = batch_size

        classes = len(np.unique(Y_train_clean))
        args.nbins = classes



        print('Num Classes: ', classes)
        ni = args.label_noise_rate
        print('+' * shutil.get_terminal_size().columns)
        print('Label noise ratio: %.3f' % ni)
        print('+' * shutil.get_terminal_size().columns)

        Y_train, mask_train = flip_label(x_train, Y_train_clean, ni, args)
        Y_valid = None
        if args.valid_set:
            Y_valid, _ = flip_label(x_valid, y_valid_clean, ni, args)
        Y_test = Y_test_clean

        train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(Y_train).long(),
                                      torch.from_numpy(np.arange(len(Y_train))), torch.from_numpy(
                Y_train_clean))  # 'Y_train_clean' is used for evaluation instead of training.

        test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                  num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.num_workers)
        valid_loader = None
        if args.valid_set:
            valid_dataset = TensorDataset(torch.from_numpy(x_valid).float(), torch.from_numpy(Y_valid).long())
            valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                      num_workers=args.num_workers)
        args.num_classes = classes
        args.seq_len = x_train.shape[2]
        args.input_size = x_train.shape[1]

        alpha_plan = [args.lr] * args.epoch
        for i in range(args.epoch_decay_start, args.epoch):
            alpha_plan[i] = float(args.epoch - i) / (args.epoch - args.epoch_decay_start) * args.lr

        model_list = []
        classifier_list = []
        loss_list = []
        loss_sample_select_list = []
        optimizer_list = []

        for s in range(args.scale_nums):
            channel = x_train.shape[1] if args.patch_type != 'before_encode' else args.embedding_size
            # Network definition
            classifier = NonLinClassifier(args.embedding_size, classes, d_hidd=args.classifier_dim,
                                          dropout=args.dropout,
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
                    classifier=classifier,
                    args=args
                ),
                args=args,
                patch_dim=args.embedding_size,
                decoder=FCNDecoder(channel, args=args) if args.recon else None,
            ).to(device)


            classifier = classifier.to(device)

            loss = build_loss(args).to(device)

            optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=args.lr)

            model_list.append(model)
            classifier_list.append(classifier)
            loss_list.append(loss)
            loss_sample_select_list.append(torch.nn.CrossEntropyLoss(reduce=False).to(device))
            optimizer_list.append(optimizer)

        up_outputs_list = []
        for _ in range(args.scale_nums):
            _output = torch.zeros(x_train.shape[0], 128).float().to(device)
            up_outputs_list.append(_output)

        test_end_five_accuracies = []
        test_end_five_f1 = []

        for epoch in range(args.epoch):
            epoch_train_loss = 0
            epoch_train_acc = 0
            num_iterations = 0

            mask_clean = []
            if epoch > args.warmup_epoch:
                for s in range(args.scale_nums):
                    model_list[s].eval()
                    classifier_list[s].eval()

                if args.small_loss_criterion == 0:  ## use jsd small loss criterion
                    JSD = calculate_scale_flow_jsd(val_loader=train_loader,
                                                   model_list=model_list,
                                                   classifier_list=classifier_list,
                                                   scale_list=args.scale_list,
                                                   num_class=args.num_classes,
                                                   num_samples=x_train.shape[0],
                                                   device=device)

                    for _jsd in JSD:
                        threshold = torch.mean(_jsd)
                        if threshold.item() > args.d_u:
                            threshold = threshold - (threshold - torch.min(_jsd)) / args.tau
                        _sr = torch.sum(_jsd < threshold).item() / x_train.shape[0]

                        _mask_clean = get_clean_class_jsd_ind(jsd_all=_jsd,
                                                              remember_rate=_sr,
                                                              class_num=args.num_classes,
                                                              target_label=Y_train)

                        mask_clean.append(_mask_clean)

            for s in range(args.scale_nums):
                model_list[s].train()
                classifier_list[s].train()
                adjust_learning_rate(alpha_plan, optimizer_list[s], epoch)

            for x, y, _, _ in train_loader:
                _start = num_iterations * args.batch_size
                _end = (num_iterations + 1) * args.batch_size
                if _end > x_train.shape[0]:
                    _end = x_train.shape[0]

                if (_end - _start) <= 1:
                    continue

                for s in range(args.scale_nums):
                    optimizer_list[s].zero_grad()

                up_scale_embed = None
                index_s = 0
                up_mask_clean = None
                pred_1 = None
                pred_embed_1 = None
                for scale_value in args.scale_list:
                    if scale_value == args.scale_list[0]:
                        pred, pred_embed = model_list[index_s](downsample_torch(x, sample_rate=scale_value, device=device), encoder = True)

                        # update features
                        if epoch > args.warmup_epoch:
                            pred_embed = args.moment_alpha * pred_embed + (1. - args.moment_alpha) * up_outputs_list[
                                                                                                         index_s][
                                                                                                     _start:_end]

                        up_scale_embed = pred_embed
                        pred = pred['bag_logits'].to(device)
                        y = y.to(device)
                        step_select_loss = loss_sample_select_list[index_s](pred, y)
                        pred_1 = pred
                        pred_embed_1 = pred_embed

                        if epoch > args.warmup_epoch:
                            if args.small_loss_criterion == 0:  ## use jsd small loss criterion
                                up_mask_clean = mask_clean[index_s][_start:_end]
                            else:
                                target_pred_label = torch.argmax(pred.data, axis=1).cpu().numpy()
                                up_mask_clean = get_clean_class_loss_ind(
                                    loss_all=step_select_loss.cpu().detach().numpy(),
                                    remember_rate=1 - rate_schedule[epoch],
                                    class_num=args.num_classes,
                                    predict_label=target_pred_label)
                        else:
                            if args.small_loss_criterion == 0:  ## use jsd small loss criterion
                                up_mask_clean = np.ones((_end - _start))
                            else:
                                up_mask_clean = get_clean_loss_ind(loss_all=step_select_loss.cpu().detach().numpy(),
                                                                   remember_rate=1 - rate_schedule[epoch])

                        up_outputs_list[index_s][_start:_end] = pred_embed.data.clone()
                    else:
                        downsample_x = downsample_torch(x, sample_rate=scale_value, device=device)

                        pred_s, pred_embed = model_list[index_s](downsample_x, encoder = True, scale=scale_value)
                        pred_s = pred_s['bag_logits']

                        if epoch > args.warmup_epoch:
                            pred_embed = args.moment_alpha * pred_embed + (1. - args.moment_alpha) * up_outputs_list[
                                                                                                         index_s][
                                                                                                     _start:_end]

                        up_scale_embed = pred_embed

                        end_knn_label_y, end_clean_mask_y = None, None

                        if epoch > args.epoch_correct_start:
                            end_knn_label, end_clean_mask = construct_lp_graph(data_embed=pred_embed,
                                                                               y_label=y,
                                                                               mask_label=up_mask_clean,
                                                                               device=device, topk=args.knn_num,
                                                                               num_real_class=args.num_classes)

                            end_knn_label_y = end_knn_label
                            end_clean_mask_y = end_clean_mask

                        if end_clean_mask_y is not None:
                            step_loss = loss_list[index_s](pred_s[end_clean_mask_y == 1],
                                                           end_knn_label_y[end_clean_mask_y == 1])
                        else:
                            step_loss = loss_list[index_s](pred_s[up_mask_clean == 1], y[up_mask_clean == 1])

                        step_loss.backward(retain_graph=True)

                        if epoch > args.warmup_epoch:
                            if args.small_loss_criterion == 0:  ## use jsd small loss criterion
                                up_mask_clean = mask_clean[index_s][_start:_end]
                            else:
                                target_pred_label = torch.argmax(pred.data, axis=1).cpu().numpy()
                                up_mask_clean = get_clean_class_loss_ind(
                                    loss_all=step_select_loss.cpu().detach().numpy(),
                                    remember_rate=1 - rate_schedule[epoch],
                                    class_num=args.num_classes,
                                    predict_label=target_pred_label)
                        else:
                            if args.small_loss_criterion == 0:  ## use jsd small loss criterion
                                up_mask_clean = np.ones((_end - _start))
                            else:
                                up_mask_clean = get_clean_loss_ind(loss_all=step_select_loss.cpu().detach().numpy(),
                                                                   remember_rate=1 - rate_schedule[epoch])

                        up_outputs_list[index_s][_start:_end] = pred_embed.data.clone()

                        if index_s == (args.scale_nums - 1):
                            epoch_train_loss += step_loss.item()

                    index_s = index_s + 1

                end_knn_label_y1, end_clean_mask_y1 = None, None
                if epoch > args.epoch_correct_start:

                    end_knn_label, end_clean_mask = construct_lp_graph(data_embed=pred_embed_1,
                                                                       y_label=y,
                                                                       mask_label=up_mask_clean,
                                                                       device=device, topk=args.knn_num,
                                                                       num_real_class=args.num_classes)

                    end_knn_label_y1 = end_knn_label
                    end_clean_mask_y1 = end_clean_mask

                if end_knn_label_y1 is not None:
                    step_loss_1 = loss_list[0](pred_1[end_clean_mask_y1 == 1], end_knn_label_y1[end_clean_mask_y1 == 1])
                else:
                    step_loss_1 = loss_list[0](pred_1[up_mask_clean == 1], y[up_mask_clean == 1])

                step_loss_1.backward(retain_graph=True)
                epoch_train_loss += step_loss_1.item()

                for s in range(args.scale_nums):
                    optimizer_list[s].step()

                num_iterations = num_iterations + 1

            epoch_train_loss = epoch_train_loss / x_train.shape[0]

            if (epoch + 10) >= args.epoch:
                for s in range(args.scale_nums):
                    model_list[s].eval()
                    classifier_list[s].eval()

                test_loss, test_accuracy, test_f1 = evaluate_scale_flow_acc(test_loader, model_list, classifier_list,
                                                                   loss_list,
                                                                   args.scale_list, device)

                test_end_five_accuracies.append(test_accuracy)
                test_end_five_f1.append(test_f1)
            print("epoch : {}, train loss: {}".format(epoch, epoch_train_loss))
            if epoch % 50 == 0:
                print("epoch : {}, train loss: {}".format(epoch, epoch_train_loss))

    endtime = time.time()
    seconds = endtime - starttime
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    deltatime = "%d:%d:%d" % (h, m, s)

    result_evalution["dataset_name"] = args.dataset
    result_evalution["avg_five_test_acc"] = round(np.mean(test_end_five_accuracies), 4)
    result_evalution["avg_five_test_f1"] = round(np.mean(test_end_five_f1), 4)
    result_evalution["deltatime"] = deltatime

    print("Training end: test_acc = ", round(np.mean(test_end_five_accuracies), 4),
          "test_f1 = ", round(np.mean(test_f1), 4),
          "traning time (seconds) = ", deltatime)

    print('Done!')

    return result_evalution




if __name__ == '__main__':
    args = parse_args()
    experiment(args)