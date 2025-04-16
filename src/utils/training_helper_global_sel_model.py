import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# torch.cuda.set_device(1)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tsaug
import wandb
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score
from src.models.FCN import FCNFeatureExtractor, FCNDecoder
from src.models.MILLET_model import (
    MILConjunctivePooling,
    GlobalAveragePooling,
    MILInstancePooling,
    MILAttentionPooling,
    MILAdditivePooling)
from src.models.MLP import MLPEncoder, MLPDecoder
from src.models.MultiTaskClassification import NonLinClassifier, MetaModel
from src.models.MILLET_SALA import MILLET as NoisyPatchModel
from src.models.model import CNNAE
from src.plot.tsne import t_sne, t_sne_during_train
from src.utils.saver import Saver
from src.utils.utils_ import readable, reset_seed_, reset_model, flip_label, remove_empty_dirs, \
    TruncatedLoss, WarmupDone
from src.utils.utils_ import select_class_by_class_with_GMM, select_with_GMM, confident_id_quality, calculate_mse_loss, \
    calculate_label_acc, combinePseudo
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

######################################################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
columns = shutil.get_terminal_size().columns


######################################################################################################


def save_model_and_sel_dict(model, args, sel_dict=None, info='', obseved_label=None):
    model_state_dict = model.state_dict()
    datestr = time.strftime(('%Y%m%d'))
    model_to_save_dir = os.path.join(args.basicpath, 'src', 'model_save', args.dataset)
    if not os.path.exists(model_to_save_dir):
        os.makedirs(model_to_save_dir, exist_ok=True)

    filename = os.path.join(model_to_save_dir, args.outfile.split('.')[0])
    if sel_dict is not None:
        filename_sel_dict = '{}{}_{}_sel_dict.npy'.format(filename, args.aug, datestr)
        np.save(filename_sel_dict, sel_dict)  # save sel_ind
    filename1 = '{}_{}_{}_{}.pt'.format(filename, args.seed, datestr, info)
    torch.save(model_state_dict, filename1)  # save model
    if obseved_label is not None:
        filename2 = '{}_{}_{}_{}.npy'.format(filename, args.seed, datestr, 'obs_label')
        np.save(filename2, obseved_label)


def test_step(data_loader, model, model2=None, args=None, epoch=None):
    model.eval()
    if model2 is not None:
        model2 = model2.eval()

    yhat = []
    ytrue = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)

            if model2 is not None:
                logits1 = model(x)
                logits2 = model2(x)
                logits = (logits1 + logits2) / 2
            else:
                if args.MILLET:
                    out_dict = model(x)
                    yhat.append(out_dict['bag_logits'].detach().cpu().numpy())
                else:
                    logits = model(x)
                    yhat.append(logits.detach().cpu().numpy())
            try:
                y = y.cpu().numpy()
            except:
                y = y.numpy()
            ytrue.append(y)

            torch.cuda.empty_cache()

    yhat = np.concatenate(yhat, axis=0)
    ytrue = np.concatenate(ytrue, axis=0)
    y_hat_proba = softmax(yhat, axis=1)
    y_hat_labels = np.argmax(y_hat_proba, axis=1)
    accuracy = accuracy_score(ytrue, y_hat_labels)
    f1_weighted = f1_score(ytrue, y_hat_labels, average='weighted')

    return accuracy, f1_weighted


def checkpoint(acc, epoch, net, args):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, args.basicpath + '/checkpoint/ckpt.t7.' +
               args.sess)


def train_model(model, train_loader, test_loader, args, train_dataset=None, saver=None, valid_loader=None):
    criterion = nn.CrossEntropyLoss(reduce=False)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160], gamma=0.5)
    early_warmup_done = WarmupDone(patience=10, args=args)
    scaler = GradScaler()
    # learning history
    train_acc_list = []
    train_acc_list_aug = []
    train_avg_loss_list = []
    test_acc_list = []
    test_f1s = []
    valid_f1s = []
    best_acc = 0.
    pseudo_acc = 0.
    pred_label_weight = args.pred_label_weight

    try:
        loss_all = np.zeros(args.num_training_samples)

        conf_num = []
        for e in range(args.epochs):
            sel_dict = {'sel_ind': [], 'lam': [], 'mix_ind': []}

            # pred_label_weight = calculate_weight(e, initial_weight=0.2, final_weight=0.9, num_epochs=args.epochs)
            # training step
            if e <= args.warmup and not early_warmup_done.warmup_done or args.just_warmup:
                if args.model in ['SALA']:
                    train_accuracy, avg_loss = warmup_global_sel_model(data_loader=train_loader,
                                                         model=model,
                                                         optimizer=optimizer,
                                                         criterion=criterion,
                                                         epoch=e,
                                                         loss_all=loss_all,
                                                         args=args,
                                                         scaler=scaler)

                sel_score = 0.
            else:
                if args.select_type == 'G_CbC_GMM':
                    confident_id, conf_score = select_class_by_class_with_GMM(loss_all=loss_all, args=args,
                                                                              labels=train_loader.dataset.tensors[1],
                                                                              p_threshold=args.p_threshold,
                                                                              model=model,
                                                                              train_loader=train_loader)
                else:
                    confident_id, conf_score = select_with_GMM(loss_all=loss_all, args=args,
                                                               labels=train_loader.dataset.tensors[1],
                                                               p_threshold=args.p_threshold,
                                                               model=model,
                                                               train_loader=train_loader)
                conf_score = torch.from_numpy(conf_score).cuda()
                sel_score = confident_id_quality(train_dataset.tensors[1], train_dataset.tensors[3], confident_id)

                train_accuracy, avg_loss, pseudo_acc = train_step(data_loader=train_loader,
                                                                  model=model,
                                                                  optimizer=optimizer,
                                                                  criterion=criterion,
                                                                  epoch=e,
                                                                  loss_all=loss_all,
                                                                  args=args,
                                                                  scaler=scaler,
                                                                  confident_id=confident_id,
                                                                  pred_label_weight=pred_label_weight,
                                                                  conf_prob=conf_score)

            if args.tsne_during_train and args.seed == args.manual_seeds[0] and e in args.tsne_epochs:
                xs, ys, _, y_clean = train_dataset.tensors
                with torch.no_grad():
                    t_sne_during_train(xs, ys, y_clean, model=model, tsne=True, args=args, sel_dict=sel_dict, epoch=e)

            # testing
            test_accuracy, f1 = test_step(data_loader=test_loader,
                                          model=model, args=args, epoch=e)

            wandb.log({'Train Acc': train_accuracy, 'Test Acc': test_accuracy, 'Test f1': f1,
                       'Sel Score': sel_score}, step=e)
            if args.label_correct_type == 'soft':
                wandb.log({'Pseudo Label Acc': pseudo_acc}, step=e)
            if args.valid_set:
                valid_accuracy, valid_f1 = test_step(data_loader=valid_loader,
                                                     model=model, args=args, epoch=e)
                valid_f1s.append(valid_f1)
                wandb.log({'Valid Acc': valid_accuracy, 'Valid f1': valid_f1}, step=e)
                print(f"Valid Acc: {valid_accuracy}, Valid f1: {valid_f1}")
                early_warmup_done(model, valid_f1)

                if early_warmup_done.warmup_done and not early_warmup_done.load_model_after_warmup:
                    early_warmup_done.load_checkpoint(model)
                scheduler.step()

            # train results each epoch
            train_acc_list.append(train_accuracy)
            # train_acc_list_aug.append(train_accuracy[1])
            train_acc_oir = train_accuracy
            train_avg_loss_list.append(avg_loss)

            # test results each epoch
            test_acc_list.append(test_accuracy)
            test_f1s.append(f1)

            print(
                '{} epoch - Train Loss {:.4f}\tTrain accuracy {:.4f}\tTest accuracy {:.4f}'.format(
                    e + 1,
                    avg_loss,
                    train_acc_oir,
                    test_accuracy))
            if args.save_model and e % 10 == 0 and args.seed == args.manual_seeds[0]:
                if best_acc < test_accuracy:
                    best_acc = test_accuracy
                    save_model_and_sel_dict(model, args, info='best')


    except KeyboardInterrupt:
        print('*' * shutil.get_terminal_size().columns)
        print('Exiting from training early')

    if args.confcsv is not None:
        csvpath = os.path.join(args.basicpath, 'src', 'bar_info')
        if not os.path.exists(csvpath):
            os.makedirs(csvpath)
        pd.DataFrame(conf_num).to_csv(os.path.join(csvpath, args.dataset + str(args.sel_method) + args.confcsv),
                                      mode='a', header=True)

    if args.save_model and args.seed == args.manual_seeds[0]:
        save_model_and_sel_dict(model, args, obseved_label=train_dataset.tensors[1].numpy())

    if args.plt_loss_hist:
        plot_train_loss_and_test_acc(train_avg_loss_list, test_acc_list, args, pred_precision=train_acc_list,
                                     aug_accs=train_acc_list_aug,
                                     saver=saver, save=True)
    if args.plot_tsne and args.seed == args.manual_seeds[0]:
        xs, ys, _, y_clean = train_dataset.tensors
        datestr = time.strftime(('%Y%m%d'))
        with torch.no_grad():
            t_sne(xs, ys, y_clean, model=model, tsne=True, args=args, datestr=datestr, sel_dict=sel_dict)

    # we test the final model at line 231.
    test_results_last_ten_epochs = dict()
    test_results_last_ten_epochs['last_ten_train_acc'] = train_acc_list[-10:]
    test_results_last_ten_epochs['last_ten_test_acc'] = test_acc_list[-10:]
    test_results_last_ten_epochs['last_ten_test_f1'] = test_f1s[-10:]
    test_results_last_ten_epochs['last_ten_valid_f1'] = valid_f1s[-10:]
    test_results_last_ten_epochs['best_valid_f1'] = np.max(valid_f1s)
    test_results_last_ten_epochs['best_valid_f1_epoch'] = np.argmax(valid_f1s)
    return model, test_results_last_ten_epochs


def train_eval_model(model, x_train, x_test, Y_train, Y_test, Y_train_clean,
                     ni, args, saver, plt_embedding=True, plt_cm=True, x_valid=None, y_valid=None):
    # if args.dataset == 'FD_A':
    #     x_train = torch.from_numpy(x_train).float().unsqueeze(1)
    #     x_test = torch.from_numpy(x_test).float().unsqueeze(1)
    #     x_valid = torch.from_numpy(x_valid).float().unsqueeze(1)

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(Y_train).long(),
                                  torch.from_numpy(np.arange(len(Y_train))), torch.from_numpy(Y_train_clean))
    ### 'Y_train_clean' is used for evaluation instead of training.

    test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(Y_test).long())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                             num_workers=args.num_workers)
    valid_loader = None
    if args.valid_set:
        valid_dataset = TensorDataset(torch.from_numpy(x_valid).float(), torch.from_numpy(y_valid).long())
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                                  num_workers=args.num_workers)

    ######################################################################################################
    # Train model

    model, test_results_last_ten_epochs = train_model(model, train_loader, test_loader, args,
                                                      train_dataset=train_dataset, saver=saver,
                                                      valid_loader=valid_loader)
    print('Train ended')

    ########################################## Eval ############################################

    test_results = dict()
    test_results['avg_last_ten_train_acc'] = np.mean(test_results_last_ten_epochs['last_ten_train_acc'])
    test_results['avg_last_ten_test_acc'] = np.mean(test_results_last_ten_epochs['last_ten_test_acc'])
    test_results['avg_last_ten_test_f1'] = np.mean(test_results_last_ten_epochs['last_ten_test_f1'])
    test_results['avg_last_ten_valid_f1'] = np.mean(test_results_last_ten_epochs['last_ten_valid_f1'])
    test_results['acc'] = test_results_last_ten_epochs['last_ten_test_acc'][-1]
    test_results['f1_weighted'] = test_results_last_ten_epochs['last_ten_test_f1'][-1]
    test_results['valid_f1'] = test_results_last_ten_epochs['last_ten_valid_f1'][-1]
    test_results['best_valid_f1'] = test_results_last_ten_epochs['best_valid_f1']
    test_results['best_valid_f1_epoch'] = test_results_last_ten_epochs['best_valid_f1_epoch']

    #############################################################################################
    plt.close('all')
    torch.cuda.empty_cache()
    return test_results


def main_wrapper_global_sel_model(args, x_train, x_test, Y_train_clean, Y_test_clean, saver, seed=None, x_valid=None,
                                  y_valid_clean=None):
    class SaverSlave(Saver):
        def __init__(self, path):
            super(Saver)
            self.args = args
            self.path = path
            self.makedir_()
            # self.make_log()

    classes = len(np.unique(Y_train_clean))
    args.nbins = classes

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
    if args.backbone == 'MLP':
        model = NoisyPatchModel(
            MLPEncoder(channel, args.patch_len, d_model=args.embedding_size),
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
            decoder=MLPDecoder(d_model=args.embedding_size, patch_len=args.patch_len) if args.recon else None,
        ).to(device)
    else: # FCN
        backbone = NoisyPatchModel
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


    nParams = sum([p.nelement() for p in model.parameters()])
    s = 'MODEL: %s: Number of parameters: %s' % ('CNN', readable(nParams))
    print(s)
    saver.append_str([s])

    ######################################################################################################
    print('Num Classes: ', classes)
    print('Train:', x_train.shape, Y_train_clean.shape, [(Y_train_clean == i).sum() for i in np.unique(Y_train_clean)])
    print('Test:', x_test.shape, Y_test_clean.shape, [(Y_test_clean == i).sum() for i in np.unique(Y_test_clean)])
    saver.append_str(['Train: {}'.format(x_train.shape),
                      'Test: {}'.format(x_test.shape), '\r\n'])

    ######################################################################################################
    # Main loop
    if seed is None:
        seed = np.random.choice(1000, 1, replace=False)

    print('#' * shutil.get_terminal_size().columns)
    print('RANDOM SEED:{}'.format(seed).center(columns))
    print('#' * shutil.get_terminal_size().columns)

    args.seed = seed

    ni = args.ni
    saver_slave = SaverSlave(os.path.join(saver.path, f'seed_{seed}', f'ratio_{ni}'))
    # True or false
    print('+' * shutil.get_terminal_size().columns)
    print('Label noise ratio: %.3f' % ni)
    print('+' * shutil.get_terminal_size().columns)

    reset_seed_(seed)
    model = reset_model(model)

    # noisy labels
    Y_train, mask_train = flip_label(x_train, Y_train_clean, ni, args)
    Y_valid = None
    if args.valid_set:
        Y_valid, _ = flip_label(x_valid, y_valid_clean, ni, args)
    Y_test = Y_test_clean

    test_results = train_eval_model(model, x_train, x_test, Y_train,
                                    Y_test, Y_train_clean,
                                    ni, args, saver_slave,
                                    plt_embedding=args.plt_embedding,
                                    plt_cm=args.plt_cm,
                                    x_valid=x_valid, y_valid=Y_valid)
    remove_empty_dirs(saver.path)

    return test_results


def plot_train_loss_and_test_acc(avg_train_losses, test_acc_list, args, pred_precision=None, saver=None, save=False,
                                 aug_accs=None):
    plt.gcf().set_facecolor(np.ones(3) * 240 / 255)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    l1 = ax.plot(avg_train_losses, '-', c='orangered', label='Training loss', linewidth=1)
    l2 = ax2.plot(test_acc_list, '-', c='blue', label='Test acc', linewidth=1)
    l3 = ax2.plot(pred_precision, '-', c='green', label='Sample_sel acc', linewidth=1)

    if len(aug_accs) > 0:
        l4 = ax2.plot(aug_accs, '-', c='yellow', label='Aug acc', linewidth=1)
        lns = l1 + l2 + l3 + l4
    else:
        lns = l1 + l2 + l3

    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper right')
    # plt.legend(handles=[l1,l2],labels=["Training loss","Test acc"],loc='upper right')

    plt.axvline(args.warmup, color='g', linestyle='--')

    ax.set_xlabel('epoch', size=18)
    ax.set_ylabel('Train loss', size=18)
    ax2.set_ylabel('Test acc', size=18)
    plt.gcf().autofmt_xdate()
    plt.title(f'Model:new model dataset:{args.dataset}')
    plt.grid(True)

    plt.tight_layout()

    saver.save_fig(fig, name=args.dataset)


def warmup_global_sel_model(data_loader, model, optimizer, criterion, epoch=None,
                            loss_all=None, args=None, scaler=None):
    global_step = 0
    avg_accuracy = 0.
    avg_loss = 0.
    model = model.train()

    MseLoss = nn.MSELoss()

    for batch_idx, (x, y_hat, x_idx, _) in enumerate(data_loader):
        if x.shape[0] == 1:
            continue
        # Forward and Backward propagation
        x, y_hat = x.to(device), y_hat.to(device)

        recon_loss = 0.
        optimizer.zero_grad()

        if args.MILLET:
            if args.recon:
                out_dict, x_recon = model(x, mode=args.mode, x_idx=x_idx, recon=True)
                if args.backbone == 'MLP':
                    recon_loss = MseLoss(x.unfold(dimension=-1, size=args.patch_len,
                                                  step=args.patch_stride), x_recon)
                else:
                    recon_loss = MseLoss(x, x_recon)
            else:
                out_dict = model(x)
            model_loss = criterion(out_dict['bag_logits'], y_hat)
            model.interpre[x_idx] = out_dict[args.interpre_type].clone().detach().cpu()
        else:
            out = model(x)
            model_loss = criterion(out, y_hat)

        loss_all[x_idx] = (1 - args.gamma) * loss_all[
            x_idx] + args.gamma * model_loss.data.detach().clone().cpu().numpy()
        model_loss = model_loss.sum() + args.L_rec_coef * recon_loss

        ############################################################################################################

        # loss exchange
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        avg_loss += model_loss.item()

        # Compute accuracy
        if args.MILLET:
            acc = torch.eq(torch.argmax(out_dict['bag_logits'], 1), y_hat).float()
        else:
            acc = torch.eq(torch.argmax(out, 1), y_hat).float()
        avg_accuracy += acc.sum().cpu().numpy()
        global_step += len(y_hat)
        torch.cuda.empty_cache()

    return avg_accuracy / global_step, avg_loss / global_step


def train_step(data_loader, model, optimizer, criterion, loss_all=None, epoch=0, args=None,
               scaler=None, confident_id=None, pred_label_weight=0.8,
               conf_prob=None):
    '''

    :param data_loader:
    :param model:
    :param optimizer:
    :param criterion:
    :param loss_all:
    :param epoch:
    :param args:
    :param scaler:
    :param confident_id: id of clean samples
    :param pred_label_weight:
    :param conf_prob: the prob of clean samples
    :return:
    '''
    global_step = 0
    aug_step = 0
    avg_accuracy = 0.
    avg_accuracy_aug = 0.
    avg_loss = 0.

    model = model.train()
    confident_set_id = np.array([])
    conf_prob = conf_prob.view(-1)
    pseudo_error = []
    pseudo_acc = []
    MseLoss = nn.MSELoss()
    KLLoss = torch.nn.KLDivLoss(reduction='batchmean')
    torch.autograd.set_detect_anomaly(True)

    for batch_idx, (x, y_hat, x_idx, y_clean) in enumerate(data_loader):
        # Forward and Backward propagation
        x, y_hat = x.to(device), y_hat.to(device)

        recon_loss = 0.

        clean_sample_mask = torch.isin(x_idx, confident_id)
        model_args = {'bags': x, 'pos': None, 'recon': args.recon, 'conf_mask': None, 'label': y_hat,
                      'forward_type': 'perturb', 'conf_score': conf_prob, 'x_idx': x_idx,
                      'only_max_min_noise': args.only_max_min_noise, 'only_max_min_mask': args.only_max_min_mask}
        # The default value of forward_type for the testing phase is "None".

        if args.recon:
            out_dict, x_recon = model(**model_args)
            model_args['forward_type'] = 'mask'
            if args.backbone == 'MLP':
                recon_loss = MseLoss(x.unfold(dimension=-1, size=args.patch_len,
                                              step=args.patch_stride), x_recon)
            else:
                recon_loss = MseLoss(x, x_recon)
                # recon_loss = MseLoss(x.unfold(dimension=-1, size=args.patch_len,
                #                               step=args.patch_stride), x_recon)
        else:
            out_dict = model(**model_args)
            model_args['forward_type'] = 'mask'

        model_loss = criterion(out_dict['bag_logits'], y_hat)
        if sum(clean_sample_mask)>0:
            conf_loss = model_loss[clean_sample_mask].sum()
        else:
            conf_loss = 0.

        consistency_loss_coef = 0.
        if args.forward_type in ['pm','mm','pp','mp','np','nm']:
            if args.consistency_loss_coef > 0:
                p1 = torch.softmax(out_dict['bag_logits'].clone().detach(), dim=1)
                p2 = torch.softmax(out_dict2['bag_logits'].clone().detach(), dim=1)
                consistency_loss_coef = MseLoss(p1,p2)

        klloss = 0.
        if args.label_correct_type != 'None':
            with torch.no_grad():
                pred_softlabel1 = torch.softmax(out_dict['bag_logits'].clone().detach(), dim=1)
                pred_softlabel2 = torch.softmax(out_dict2['bag_logits'].clone().detach(), dim=1)
                pred_softlabel1 = (pred_softlabel1 + pred_softlabel2) / 2

                probs = conf_prob[x_idx][~clean_sample_mask].float()
                probs = probs.clamp(min=0.01, max=0.99).to(device).view(-1, 1)

                pred_pseudo = probs * F.one_hot(y_hat, num_classes=args.nbins).float()[~clean_sample_mask] + \
                              (1 - probs) * pred_softlabel1[~clean_sample_mask]
                max_probs, _ = torch.max(pred_pseudo, dim=1)
                noisy_sample_indices = torch.where(~clean_sample_mask)[0]
                selected_indices = max_probs.cpu() > args.correct_threshold
                noisy_sample_indices = noisy_sample_indices[selected_indices]
                selected_pseudo_labels = pred_pseudo[selected_indices]
            if len(noisy_sample_indices) > 0:
                klloss = KLLoss(F.log_softmax(out_dict['bag_logits'][noisy_sample_indices], dim=-1),
                                selected_pseudo_labels)

        Tloss = conf_loss + args.pseudo_loss_coef * klloss + args.L_rec_coef * recon_loss + \
                args.consistency_loss_coef * consistency_loss_coef


        # loss exchange
        optimizer.zero_grad()
        Tloss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        avg_loss += Tloss.item()

        # Compute accuracy
        if args.MILLET:
            acc1 = torch.eq(torch.argmax(out_dict['bag_logits'], 1), y_hat).float()
        else:
            acc1 = torch.eq(torch.argmax(out, 1), y_hat).float()
        avg_accuracy += acc1.sum().cpu().numpy()

        global_step += len(y_hat)

    return avg_accuracy / global_step, avg_loss / global_step, \
        sum(pseudo_acc) / len(pseudo_acc) if len(pseudo_acc) != 0 else 0
