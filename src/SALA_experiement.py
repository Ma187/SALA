import argparse
import logging
import os
import sys
import time

sys.path.append(os.path.dirname(sys.path[0]))
import shutil
import wandb
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyts import datasets

torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True
from datetime import datetime
# sys.path.append("..")

from src.utils.log_utils import StreamToLogger, get_logger, create_logfile
from src.utils.utils_ import create_synthetic_dataset
from src.utils.global_var import OUTPATH
from src.utils.saver import Saver
from src.utils.training_helper_single_model import main_wrapper_single_model
from src.utils.training_helper_global_sel_model import main_wrapper_global_sel_model

from src.utils.training_helper_sanm import main_wrapper_global_sel_model as main_wrapper_sanm
from src.utils.utils_scale import build_dataset_pt

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
columns = shutil.get_terminal_size().columns


######################################################################################################

def parse_args(comstr=None):
    # TODO: make external configuration file -json or similar.
    """
    Parse arguments
    """

    # Add global parameters
    parser = argparse.ArgumentParser(description='')

    # Synth Data
    parser.add_argument('--dataset', type=str, default='', help='') # choices=['SleepEDF','FD_A']
    parser.add_argument('--outfile', type=str, default='CTW.csv', help='name of output file')
    parser.add_argument('--ni', type=float, default=0.5, help='label noise ratio')
    parser.add_argument('--label_noise', type=int, default=0, help='Label noise type, sym or int for asymmetric, '
                                                                   'number as str for time-dependent noise')

    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--noise_type', default='asymmetric', help='symmetric or asymmetric')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of in-distribution classes')




    parser.add_argument('--_batch_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_gradual', type=int, default=100)

    parser.add_argument('--normalization', type=str, default='batch')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--l2penalty', type=float, default=1e-4)

    parser.add_argument('--num_workers', type=int, default=0, help='PyTorch dataloader worker. Set to 0 if debug.')
    parser.add_argument('--seed', type=int, default=0, help='RNG seed - only affects Network init')

    parser.add_argument('--classifier_dim', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=128)

    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--filters', nargs='+', type=int, default=[128, 128, 256, 256])
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--padding', type=int, default=2)

    # Suppress terminal out
    parser.add_argument('--disable_print', action='store_true', default=False)
    parser.add_argument('--plt_embedding', action='store_true', default=False)
    parser.add_argument('--plt_loss_hist', action='store_true', default=False)
    parser.add_argument('--plt_cm', action='store_true', default=False)
    parser.add_argument('--plt_recons', action='store_true', default=False, help='plot AE reconstructions')
    parser.add_argument('--headless', action='store_true', default=False, help='Matplotlib backend')
    parser.add_argument('--warmup', type=int, default=20, help='warmup epochs')

    parser.add_argument('--model', choices=['single_ae_aug_after_sel', 'single_aug', 'single_sel', 'vanilla',
                                            'single_aug_after_sel', 'single_ae_sel', 'single_ae', 'single_ae_aug',
                                            'single_ae_aug_sel_allaug', 'single_ae_aug_before_sel',
                                            'SALA','sanm'], default='SALA')

    parser.add_argument('--manual_seeds', type=int, nargs='+', default=[37, 118, 337, 815, 19],
                        # For fair comparation, we set the same seeds for all methods.
                        help='manual_seeds for five folds cross varidation')
    parser.add_argument('--label_correct_type', type=str, default='None', choices=['None', 'hard', 'soft'],
                        help='if correct label')
    parser.add_argument('--num_training_samples', type=int, default=0, help='num of trainging samples')
    parser.add_argument('--mean_loss_len', type=int, default=1, help='the length of mean loss')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='the weight of current sample loss in mean_loss_sel method')
    parser.add_argument('--arg_interval', type=int, default=1,
                        help='the batch-interval for augmentation in batch')
    parser.add_argument('--cuda_device', type=int, default=0, help='choose the cuda devcie')
    parser.add_argument('--aug',
                        choices=['GNoise', 'NoAug', 'Oversample', 'Convolve', 'Crop', 'Drift', 'TimeWarp', 'Mixup'],
                        default='NoAug')
    parser.add_argument('--sample_len', type=int, default=0)
    parser.add_argument('--ucr', type=int, default=0, help='if 128, run all ucr datasets')
    parser.add_argument('--basicpath', type=str, default='', help='basic path')
    parser.add_argument('--plot_tsne', action='store_true', default=False, help='if plot t-sne or not')
    parser.add_argument('--nbins', type=int, default=0, help='number of class')
    parser.add_argument('--save_model', action='store_true', default=False, help='if save model or not')
    parser.add_argument('--from_ucr', type=int, default=0, help='begin from which dataset')
    parser.add_argument('--end_ucr', type=int, default=128, help='end at which dataset')
    parser.add_argument('--sel_method', type=int, default=3, choices=[0, 1, 2, 3, 4],
                        help='''0: select ratio is known (co-teaching, sigua);
                                1,2: select confident samples class by class;
                                3: select w/ EPS
                                4: select w/o EPS''')
    parser.add_argument('--tsne_during_train', action='store_true', default=False,
                        help='if plot tsne during training or not')
    parser.add_argument('--tsne_epochs', type=int, nargs='+', default=[49, 99, 149, 199, 249, 299],
                        help='manual_seeds for five folds cross varidation')

    parser.add_argument('--augMSE', action='store_true', default=False, help='if use MSE on aug or not')
    parser.add_argument('--bad_weight', type=float, default=1e-3, help='for sigua')
    parser.add_argument('--aug_ae', action='store_true', default=False, help='if reconstruct augmented samples or not')
    parser.add_argument('--L_aug_coef', type=float, default=1.,
                        help='the coefficient of L_aug')
    parser.add_argument('--L_rec_coef', type=float, default=1.,
                        help='the coefficient of L_rec')
    parser.add_argument('--confcsv', type=str, default=None,
                        help='the file of saving conf_num')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    parser.add_argument('--standardization_choice', type=str, choices=['z-score', 'min-max'], default='z-score',
                        help='choose the method of standardization')
    parser.add_argument('--debug', action='store_true', default=False, help='')
    parser.add_argument('--valid_set', action='store_true', default=False, help='')

    parser.add_argument('--sess', default='default', type=str, help='session id')
    parser.add_argument('--start_prune', default=40, type=int,
                        help='')

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
    parser.add_argument('--interpre_type', type=str, default='interpretation',
                        choices=['interpretation', 'atten'], help='')

    ## wandb
    parser.add_argument('--group', type=str, default='my group')
    parser.add_argument('--exp_name', type=str, default='my exp')
    parser.add_argument('--use_wandb', action='store_true', default=False)

    parser.add_argument('--patch', action='store_true', default=False)
    parser.add_argument('--correct_threshold', type=float, default=0.7)
    parser.add_argument('--show_shapelets_tsne_epochs', type=int, default=5000)
    parser.add_argument('--project', type=str, default='Shapelets')

    parser.add_argument('--pred_label_weight', type=float, default=0.8)
    parser.add_argument('--pseudo_label', type=str, default='none', choices=['pred', 'pred_shapelets', 'none'])
    parser.add_argument('--patch_type', type=str, default='before_encode', choices=['before_encode', 'after_encode'])
    parser.add_argument('--recon', action='store_true', default=False)
    parser.add_argument('--soft_weight', action='store_true', default=False)
    parser.add_argument('--temperature', type=float, default=1.)

    parser.add_argument('--forward_type', default='None',type=str,choices=['None','perturb','mask',
                                                                           'random_mask','pm','mm','pp','mp'])
    parser.add_argument('--only_max_min_noise', default='None',type=str,choices=['None','max_min','max','near_max_min'])
    parser.add_argument('--only_max_min_mask', default='None',type=str,choices=['None','max_min','max','near_max_min'])
    parser.add_argument('--amp_noise', type=float, default=0.1)
    parser.add_argument('--amp_mask', type=float, default=0.1)
    
    parser.add_argument('--_patch_len', type=int, default=8, help='predefined args')
    parser.add_argument('--patch_len', type=int, default=8, help='')
    parser.add_argument('--_patch_stride', type=float, default=1, help='predefined args')
    parser.add_argument('--patch_stride', type=int, default=1, help='')
    parser.add_argument('--_len_shapelet', type=float, nargs='+', default=[0.05], help="predefined args")
    parser.add_argument('--len_shapelet', type=int, nargs='+', default=[0.05], help="patch length")
    parser.add_argument('--_shapelet_stride', type=float, default=0.5, help="predefined args")
    parser.add_argument('--shapelet_stride', type=int, default=0.5, help="stride")
    parser.add_argument('--backbone', type=str, default="FCN", choices=['FCN','MLP'])
    parser.add_argument('--mean_norm', type=int, default=0)
    parser.add_argument('--consistency_loss_coef', type=float, default=0.)
    parser.add_argument('--pseudo_loss_coef', type=float, default=1.)
    parser.add_argument('--nvars', type=int, default=1)
    parser.add_argument('--consistency_loss', default='MSE', type=str, choices=['JS','MSE'])
    parser.add_argument('--perturbPred', default=False, action="store_true")
    parser.add_argument('--score_gap', default=0.02,type=float)
    parser.add_argument('--sanm', default=False, action="store_true")
    parser.add_argument('--smooth_beta', type=float, default=0.01, help='weight of current value')

    # Add parameters for each particular network

    if comstr is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(comstr.split(' '))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device('cuda'):
        torch.cuda.set_device(args.cuda_device)
    return args


######################################################################################################
def main(args, dataset_name=None):
    saver = Saver(OUTPATH, os.path.basename(__file__).split(sep='.py')[0],
                  hierarchy=os.path.join(args.dataset), args=args)

    if args.plot_tsne:
        args.save_model = True

    ######################################################################################################
    print(f'{args}')

    ######################################################################################################
    SEED = args.seed
    # TODO: implement multi device and different GPU selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    ######################################################################################################
    # Data
    print('*' * shutil.get_terminal_size().columns)
    print('Dataset: {}'.format(args.dataset).center(columns))
    print('*' * shutil.get_terminal_size().columns)

    five_test_acc = []
    five_test_f1 = []
    five_avg_last_ten_test_acc = []
    five_avg_last_ten_test_f1 = []

    result_evalution = dict()

    train_dataset, train_target, test_dataset, test_target, classes, val_dataset, val_target = build_dataset_pt(args)
    args.nbins = classes

    args.patch_len = max(min(int(args.sample_len * args._len_shapelet[0]), args._patch_len),1)  # MILLET_model.py , FCNDecoder 和 shapelet_utils.py中stride修改

    args.shapelet_stride = max(int(args.patch_len * args._shapelet_stride), 1)

    args.patch_stride = args.shapelet_stride

    id_acc = 0
    seeds = args.manual_seeds
    starttime = time.time()

    current_date = datetime.now().date()
    current_date = current_date.strftime("%Y%m%d")


    if args.use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=args.project,
            group=current_date + args.group + f"_{args.dataset}_Len{args._len_shapelet}_S{args.shapelet_stride}_PL{args.patch_len}",
            # track hyperparameters and run metadata
            config=args,
        )
    else:
        wandb.init(mode="disabled")

    id_acc = id_acc + 1
    x_train = train_dataset
    x_test = test_dataset
    Y_train_clean = train_target
    Y_test_clean = test_target

    x_valid = None
    y_valid_clean = None
    if args.valid_set:
        x_valid = val_dataset
        y_valid_clean = val_target

    print("id_acc = ", id_acc, Y_train_clean.shape, Y_test_clean.shape)
    print('\nx_train shape\n', x_train.shape)
    args.num_classes = classes
    args.num_training_samples = len(x_train)
    Y_train_clean_ = Y_train_clean.astype(int)
    args.k_val = min(np.median(np.bincount(Y_train_clean_)).astype(int), args.k_val)


    batch_size = min(x_train.shape[0] // 10, args._batch_size)
    if x_train.shape[0] % batch_size == 1:
        batch_size += -1
    print(f'Batch size: {batch_size}')
    args.batch_size = batch_size
    args.test_batch_size = batch_size

    # ##########################
    # ##########################
    saver.make_log(**vars(args))
    ######################################################################################################
    for seed in seeds:
        if args.model in ['vanilla']:
            df_results = main_wrapper_single_model(args, x_train, x_test, Y_train_clean, Y_test_clean, saver,
                                                   x_valid=x_valid, y_valid_clean=y_valid_clean)
        elif args.model in ['sanm']:
            df_results = main_wrapper_sanm(args, x_train, x_test, Y_train_clean, Y_test_clean, saver,
                                           x_valid=x_valid, y_valid_clean=y_valid_clean, seed=seed)
        else: # ['SALA']:
            df_results = main_wrapper_global_sel_model(args, x_train, x_test, Y_train_clean, Y_test_clean, saver,
                                                           x_valid=x_valid, y_valid_clean=y_valid_clean,seed=seed)

    five_test_acc.append(df_results["acc"])
    five_test_f1.append(df_results["f1_weighted"])
    five_avg_last_ten_test_acc.append(df_results["avg_last_ten_test_acc"])
    five_avg_last_ten_test_f1.append(df_results["avg_last_ten_test_f1"])

    wandb.finish()

    endtime = time.time()
    result_evalution["dataset_name"] = args.dataset
    result_evalution["avg_five_test_acc"] = round(np.mean(five_test_acc), 4)
    result_evalution["std_five_test_acc"] = round(np.std(five_test_acc), 4)
    result_evalution["avg_five_test_f1"] = round(np.mean(five_test_f1), 4)
    result_evalution["std_five_test_f1"] = round(np.std(five_test_f1), 4)
    result_evalution["avg_five_avg_last_ten_test_acc"] = round(np.mean(five_avg_last_ten_test_acc), 4)
    result_evalution["std_five_avg_last_ten_test_acc"] = round(np.std(five_avg_last_ten_test_acc), 4)
    result_evalution["avg_five_avg_last_ten_test_f1"] = round(np.mean(five_avg_last_ten_test_f1), 4)
    result_evalution["std_five_avg_last_ten_test_f1"] = round(np.std(five_avg_last_ten_test_f1), 4)


    seconds = endtime - starttime
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    deltatime = "%d:%d:%d" % (h, m, s)
    result_evalution["deltatime"] = deltatime
    return result_evalution


######################################################################################################
# if __name__ == '__main__':

def experiment(args):
    # args = parse_args()
    if args.use_wandb:
        wandb.login(key="..")

    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    basicpath = os.path.dirname(father_path)

    # Logging setting
    if not args.debug:  # if not debug, no log.
        logger = get_logger(logging.INFO, args.debug, args=args, filename='logfile.log')
        __stderr__ = sys.stderr  #
        sys.stderr = open(create_logfile(args, 'error.log'), 'a')
        __stdout__ = sys.stdout
        sys.stdout = StreamToLogger(logger, logging.INFO)

    print(f"father_path = {father_path}")
    result_value = []

    other_dataset_list = ['SleepEDF', 'FD_A']

    if args.dataset == '':
        for dataset_name in other_dataset_list:
            args = parse_args()
            args.basicpath = basicpath
            args.dataset = dataset_name

            df_results = main(args, dataset_name)
            result_value.append(df_results)

            print(f'result_value = {result_value}')

            path = os.path.abspath(os.path.join(basicpath, 'statistic_results'))
            os.makedirs(path,exist_ok=True)
            path = os.path.abspath(os.path.join(path,args.outfile))
            print(path)
            pd.DataFrame(result_value).to_csv(path)
    else:
        args.basicpath = basicpath
        df_results = main(args, args.dataset)
        result_value.append(df_results)
        print(f'result_value = {result_value}')

        path = os.path.abspath(os.path.join(basicpath, 'statistic_results', args.outfile))
        print(path)
        pd.DataFrame(result_value).to_csv(path)

    return result_value



if __name__ == '__main__':
    args = parse_args()
    experiment(args)
