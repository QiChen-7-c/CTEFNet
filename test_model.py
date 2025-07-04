import random

import numpy as np
import argparse
from copy import deepcopy

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import torch
from torch.utils.data import DataLoader

from utils import ENSODataloader, load_yaml_config
from network import init_model


seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

mpl.use("Agg")
plt.rc("font", family="Arial")
mpl.rc("image", cmap="RdYlBu_r")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"


def get_args():
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--config_file_path', type=str, default='./configs/CTEFNet_test.yaml')
    parser.add_argument('--class_id', type=int, default=-1, help='class id')
    parser.add_argument('--output_dir', type=str, default='./summary/sensitivity', help='output directory to save results')
    params = parser.parse_args()
    return params


def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".pkl":
                L.append(os.path.join(root, file))
    return L


def cal_ninoskill2(pre_nino_all, real_nino):
    """
    :param pre_nino_all: [n_yr,start_mon,lead_max]
    :param real_nino: [n_yr,12]
    :return: nino_skill: [12,lead_max]
    """
    lead_max = pre_nino_all.shape[2]
    nino_skill = np.zeros([12, lead_max])
    for ll in range(lead_max):
        lead = ll + 1
        dd = deepcopy(pre_nino_all[:, :, ll])
        for mm in range(12):
            bb = dd[:, mm]
            st_m = mm + 1
            terget = st_m + lead
            if 12 < terget <= 24:
                terget = terget - 12
            elif terget > 24:
                terget = terget - 24
            aa = deepcopy(real_nino[:, terget - 1])
            nino_skill[mm, ll] = np.corrcoef(aa, bb)[0, 1]
    return nino_skill


def func_pre(cfg):
    data_params = cfg.get('data')
    model_params = cfg.get('model')

    if model_params.get('name') in ['Geoformer', 'SimVP', 'SimVPV2', 'SimVPIT', 'DiffCast']:
        data_type = 'fig'
    elif model_params.get('name') in ['CGformer']:
        data_type = 'nino_fig'
    else:
        data_type = 'index'

    valid_dataset = ENSODataloader(
        data_params.get('predictor'),
        data_params.get('predictand'),
        data_params.get('valid_models'),
        data_params.get('valid_period'),
        data_params.get('obs_time'),
        data_params.get('pred_type'),
        data_params.get('pred_time'),
        data_params.get('input_region'),
        data_params.get('target_region'),
        output_type=data_type,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=data_params.get('valid_batch_size'),
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    model_params = cfg.get('model')
    model_params.update({'load_pretrain': True})
    model = init_model(cfg)
    model = model.cuda()
    pred_list = []
    true_list = []

    model.eval()
    pred_list = []
    true_list = []
    with torch.no_grad():
        if model_params.get('name') in ['CTEFNet']:
            for batch_idx, (inputs, output, index) in enumerate(valid_dataloader):
                inputs = inputs.cuda()
                index = index.cuda()
                pred, _, _ = model(inputs, None)

                pred_list.append(get_nino34(cfg, pred).detach().cpu())
                true_list.append(get_nino34(cfg, index).detach().cpu())
        else:
            for batch_idx, (inputs, output, index) in enumerate(valid_dataloader):
                inputs = inputs.cuda()
                index = index.cuda()
                pred = model(inputs)

                pred_list.append(get_nino34(cfg, pred).detach().cpu())
                true_list.append(get_nino34(cfg, index).detach().cpu())
        pred = torch.cat(pred_list, dim=0).numpy()
        true = torch.cat(true_list, dim=0).numpy()
        score, leading_mon = callback(cfg, pred[:, :], true[:, :])


def get_nino34(cfg, x):
    model_params = cfg.get('model')
    data_params = cfg.get('data')
    if model_params.get('name') == 'Geoformer':
        return x[
               :,
               :,
               0,
               data_params.get('target_region')[0]-data_params.get('input_region')[0]: data_params.get('target_region')[1]-data_params.get('input_region')[0],
               data_params.get('target_region')[2]-data_params.get('input_region')[2]: data_params.get('target_region')[3]-data_params.get('input_region')[2]
               ].mean(axis=(2,3))
    elif model_params.get('name') in ['CGformer']:
        return x.mean(axis=-1)
    else:
        return x

def callback(cfg, pred, true):
    data_params = cfg.get('data')
    pred_ = pred - np.mean(pred, axis=0)
    true_ = true - np.mean(true, axis=0)
    corr_array = (pred_ * true_).sum(axis=0) / (
        np.sqrt(np.sum(pred_ ** 2, axis=0) * np.sum(true_ ** 2, axis=0)) + 1e-6)
    rmse_array = np.sqrt(np.mean((pred - true) ** 2, axis=0))
    mae_array = np.mean(np.abs(pred - true), axis=0)
    corr = np.mean(corr_array)
    rmse = np.sum(np.sqrt(np.mean((pred - true) ** 2, axis=0)))

    if data_params.get('pred_type')=='series':
        ninoweight = (np.array([1.5] * 6 + [2] * 6 + [3] * 6 + [4] * 6 + [5] * 12) * np.log(np.arange(36) + 2))
        accskill = np.sum(ninoweight[:data_params.get('pred_time')] * corr_array)
        score = 2/3.0 * accskill - rmse
        leading_mon = len(corr_array[corr_array >= 0.5])
        print('Score: {:.4f}, Corr: {:.4f}, MSE: {:.4f}, Leading Month: {:.0f}'.format(score, corr, rmse, leading_mon))
        print('Pred Corr: {}'.format(np.round(corr_array, 4)))
    else:
        score, leading_mon, corr_list = corr, None, None
    return score, leading_mon



if __name__ == '__main__':
    args = get_args()
    config = load_yaml_config(args.config_file_path)
    func_pre(config)
    