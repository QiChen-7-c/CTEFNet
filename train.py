import os
import random

import argparse
import numpy as np
# import scipy.stats as sps
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torchvision import transforms

from utils import RandomChannelErasing, RandomFrameErasing, AverageMeter
from utils import init_dataloader, LRwarmup, load_yaml_config, make_dir, get_logger
from network import init_model, get_loss

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--config_file_path', type=str, default='./configs/CTEFNet_pretrain.yaml')
    params = parser.parse_args()
    return params


def train_epoch(cfg, epoch, model, train_dataloader, eval_dataloader, criterion, optimizer, lr_scheduler, log):
    losses = AverageMeter()
    scaler = GradScaler()
    summary_params = cfg.get('summary')
    opt_params = cfg.get('optimizer')
    data_len = len(train_dataloader)
    mseloss = nn.MSELoss()
    for batch_idx, (inputs, outputs, index) in enumerate(train_dataloader):
        model.train()
        lr_scheduler.step()
        optimizer.zero_grad()
        inputs = inputs.cuda()
        outputs = outputs.cuda()
        index = index.cuda()
        with autocast():        
            pred, tgt_em, pred_em = model(inputs, outputs)     
            loss = criterion(pred, index)
            loss += mseloss(pred_em, tgt_em) * 1
            regularization_loss = model.module.input_reg.regularization() * opt_params.get('penalty')
            loss += regularization_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.update(loss.item())
        if (batch_idx + 1) % summary_params.get('print_freq') == 0:
            log.info(' Train: [{0}][{1}/{2}]\t'
                     'lr ({lr0:.6f}, {lr1:.6f}) \t loss {loss.val:.3f} ({loss.avg:.3f})\t'
                     'reg {rm:.4f}'.format( epoch, batch_idx + 1, data_len,
                                                        lr0=optimizer.state_dict()['param_groups'][0]['lr'],
                                                        lr1=optimizer.state_dict()['param_groups'][1]['lr'],
                                                        loss=losses, rm=regularization_loss))

        if (batch_idx + 1) % summary_params.get('eval_freq') == 0 and (epoch+1 >=1):
            result, mon = test_epoch(cfg, epoch, model, eval_dataloader, criterion, log)

            if epoch >= 0:
                save_checkpoints(cfg, model, result, mon, log)

                if summary_params.get('current_patient') >= summary_params.get('patient'):
                    log.info(
                        'early stop with best score: {} , leading {} months'.format(summary_params.get('best_result'),
                                                                                    summary_params.get('leading_mon')))
                    return True

    return False

def test_epoch(cfg, epoch, model, data_loader, criterion, log):
    data_params = cfg.get('data')
    predictands = data_params.get('predictand')
    score_list = []
    leading_mon_list = []

    model.eval()
    eval_losses = AverageMeter()
    pred_list = []
    true_list = []
    with torch.no_grad():
        for batch_idx, (inputs, output, index) in enumerate(data_loader):
            inputs = inputs.cuda()
            index = index.cuda()
            pred, _, _ = model(inputs, None)
            eval_loss = criterion(pred, index).item()
            eval_losses.update(eval_loss)

            pred_list.append(get_nino34(cfg, pred).detach().cpu())
            true_list.append(get_nino34(cfg, index).detach().cpu())
        pred = torch.cat(pred_list, dim=0).numpy()
        true = torch.cat(true_list, dim=0).numpy()
        log.info(' Eval: [{}] loss {:.3f}'.format(epoch, eval_losses.avg))
        for i, predictand in enumerate(predictands):
            log.info('Eval for Index: {}'.format(predictand))
            score,leading_mon = callback(cfg, pred[:, i, :], true[:, i, :], log)
            score_list.append(score)
            leading_mon_list.append(leading_mon)
    return np.mean(score_list),  np.mean(leading_mon_list)


def get_nino34(cfg, x):
    model_params = cfg.get('model')
    data_params = cfg.get('data')
    if model_params.get('name') == 'Geoformer':
        index = x[
        :,
        :,
        0,
        data_params.get('target_region')[0] - data_params.get('input_region')[0]: data_params.get('target_region')[1] -
                                                                                  data_params.get('input_region')[0]+1,
        data_params.get('target_region')[2] - data_params.get('input_region')[2]: data_params.get('target_region')[3] -
                                                                                  data_params.get('input_region')[2]+1
        ].mean(axis=(2, 3))
        return index[:, None, :]
    elif model_params.get('name') in ['CGformer']:
        return x.mean(axis=-1)
    else:
        return x


def callback(cfg, pred, true, log):
    data_params = cfg.get('data')
    pred_ = pred - np.mean(pred, axis=0)
    true_ = true - np.mean(true, axis=0)
    corr_array = (pred_ * true_).sum(axis=0) / (
        np.sqrt(np.sum(pred_ ** 2, axis=0) * np.sum(true_ ** 2, axis=0)) + 1e-6)
    corr = np.mean(corr_array)
    rmse = np.sum(np.sqrt(np.mean((pred - true) ** 2, axis=0)))

    if data_params.get('pred_type')=='series':
        ninoweight = (np.array([1.5] * 6 + [2] * 6 + [3] * 6 + [4] * 6 + [5] * 12) * np.log(np.arange(36) + 2))
        accskill = np.sum(ninoweight[:data_params.get('pred_time')] * corr_array)
        score = 2/3.0 * accskill - rmse
        # score = accskill - rmse
        leading_mon = len(corr_array[corr_array >= 0.5])
        log.info('Score: {:.4f}, Corr: {:.4f}, MSE: {:.4f}, Leading Month: {:.0f}'.format(score, corr, rmse, leading_mon))
        log.info('Pred Corr: {}'.format(np.round(corr_array, 4)))
    else:
        score, leading_mon, corr_list = corr, None, None
        log.info('Corr: {:.4f}, MSE: {:.4f}'.format(corr, rmse))
    return score, leading_mon


def save_checkpoints(cfg, model, eval_result, leading_mon, log):
    model_params = cfg.get('model')
    data_params = cfg.get('data')
    summary_params = cfg.get('summary')
    if eval_result > summary_params.get('best_result'):
        summary_params.update({'best_result': eval_result})
        summary_params.update({'current_patient': 0})
        summary_params.update({'leading_mon': leading_mon})

        save_dir = os.path.join(summary_params.get('summary_dir'), model_params.get('name'), ''.join(data_params.get('predictand')))
        make_dir(save_dir)
        torch.save(model.module.state_dict(),
                   os.path.join(save_dir, model_params.get('name') + '_' + model_params.get('mode')  + str(summary_params.get('stage'))+'.ckpt'))
        log.info('update checkpoints, best score: {:.6f}'.format(summary_params.get('best_result')))
    else:
        summary_params.update({'current_patient': summary_params.get('current_patient')+1})
        log.info('patient index: {}'.format(summary_params.get('current_patient')))


if __name__=="__main__":
    print("pid: {}".format(os.getpid()))
    args = get_args()

    config = load_yaml_config(args.config_file_path)

    model_params = config.get('model')
    data_params = config.get('data')
    optim_params = config.get('optimizer')
    summary_params = config.get('summary')
    make_dir(os.path.join(summary_params.get('summary_dir'), model_params.get('name')))

    logger = get_logger(config)

    logger.info('Loading dataset...')
    if data_params.get('channel_erasing') != 0 and data_params.get('frame_erasing') != 0:
        train_transform = transforms.Compose([
            RandomChannelErasing(data_params.get('channel_erasing')),
            RandomFrameErasing(data_params.get('frame_erasing')),
        ])
    else:
        train_transform = None
    train_dataloader, eval_dataloader = init_dataloader(config, train_transform)
    logger.info('Dataset loaded')

    model = init_model(config)
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    logger.info('Training {} model'.format(model_params.get('name')))

    params1 = [p for n, p in model.named_parameters() if 'input_reg' in n]
    params2 = [p for n, p in model.named_parameters() if 'input_reg' not in n]
    optimizer = torch.optim.Adam([
        {"params": params1, "weight_decay": 0},
        {"params": params2, "weight_decay": 0.0001},
    ], lr=0.0001)
    lr_scheduler = LRwarmup(
        optimizer,
        lr_max=optim_params.get('lr_max'),
        lr_min=optim_params.get('lr_min'),
        warm_milestone=optim_params.get('warm_milestone'),
        annealing_index=optim_params.get('annealing_index'),
    )
    criterion = get_loss(config)

    # 开始训练
    for epoch in range(optim_params.get('epoch')):
        early_stop = train_epoch(config, epoch, model, train_dataloader, eval_dataloader, criterion, optimizer, lr_scheduler, logger)
        if early_stop:
            break

