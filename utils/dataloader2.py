import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch.utils.data import DataLoader


def init_dataloader(cfg, transforms=None):
    model_params = cfg.get('model')
    data_params = cfg.get('data')

    if model_params.get('name') in ['Geoformer', 'SimVP', 'SimVPV2', 'SimVPIT', 'DiffCast']:
        data_type = 'fig'
    elif model_params.get('name') in ['CGformer']:
        data_type = 'nino_fig'
    else:
        data_type = 'index'

    train_dataset = ENSODataloader(
        data_params.get('predictor'),
        data_params.get('predictand'),
        data_params.get('train_models'),
        data_params.get('train_period'),
        data_params.get('obs_time'),
        data_params.get('pred_type'),
        data_params.get('pred_time'),
        data_params.get('input_region'),
        data_params.get('target_region'),
        output_type=data_type,
        transform=transforms
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_params.get('train_batch_size'),
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

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

    return train_dataloader, valid_dataloader



def runmean(data, n_run):
    ll = data.shape[0]
    data_run = np.zeros([ll])
    for i in range(ll):
        if i < (n_run - 1):
            data_run[i] = np.nanmean(data[0: i + 1])
        else:
            data_run[i] = np.nanmean(data[i - n_run + 1: i + 1])
    return data_run


def load_index(root, predictands, in_models):
    uni = []
    if not predictands:
        return uni
    for model in in_models:
        for predictand in predictands:
            file_list = os.listdir(os.path.join(root, predictand))
            file_list.sort()
            var_sep = []
            for file in file_list:
                model_name = file.split('_')[0]
                mode_name = file.split('_')[1]
                if model_name == 'GODAS' or model_name == 'ORAS5' or mode_name == 'ssp370':
                    if model_name == model:
                        model_data = np.load(os.path.join(root, predictand, file))['data']
                        model_data = runmean(model_data, 3)                    
                        var_sep.append(model_data[None, :])
            uni.append(var_sep[0])
    return uni


def union_var(root: str, var_list: list, in_models=None):
    data = []
    for model in in_models:
        vdata = []
        for var in var_list:
            if var.endswith('_mm'):
                load_type = 'mean_map'
                var = var.rstrip('_mm')
            else:
                load_type = 'data'
            file_list = os.listdir(os.path.join(root, var))
            file_list.sort()
            # print(file_list)
            var_sep = []
            for file in file_list:
                # print(file)
                model_name = file.split('_')[0]
                mode_name = file.split('_')[1]

                if model_name == 'GODAS' or model_name == 'ORAS5' or mode_name == 'ssp370':
                    if model_name != model:
                        continue
                    model_data = np.load(os.path.join(root, var, file))[load_type]
                    model_data = np.nan_to_num(model_data)
                    model_data[abs(model_data) > 999] = 0
                    # print(model_data[:, None].shape)
                    var_sep.append(model_data[:, None])

            vdata.append(var_sep[0])

        if len(vdata) == 0:
            continue
        vdata = np.concatenate(vdata, axis=1)
        data.append(vdata)
        print("Loading model: ", model, vdata.shape)
    return data
############################################################################################################################################


class ENSODataloader(Dataset):
    def __init__(self, in_channels, out_channels, in_models, time_range, obs_time, pred_type, pred_time, input_region, target_region, output_type='index', transform=None):
        self.obs_time = obs_time
        self.pred_time = pred_time
        self.pred_type = pred_type
        self.output_type = output_type
        self.input_region = input_region
        self.target_region = target_region
        self.transform = transform
        
        if 'GODAS' in in_models:
            file_path = "/mnt/disk1/ctefnet_data/ReanalysisVar/GODAS/"
            full_time = pd.date_range(start='19800101', end='20231201', freq="MS")
            godas_channel_dict = {
                'tos': 'pottmp_5', 'thetao_5': 'pottmp_5', 'thetao_20': 'pottmp_20', 'thetao_40': 'pottmp_40',
                'thetao_60': 'pottmp_60', 'thetao_90': 'pottmp_90', 'thetao_120': 'pottmp_120', 'thetao_150': 'pottmp_150',
                'uo_5': 'ucur_5', 'vo_5': 'vcur_5', 'wo_10': 'dzdt_10',
                'tauuo': 'uflx', 'tauvo': 'vflx', 'zos': 'sshg', 'vor': 'vor',
                'hfds': 'thflx', 'thetao_wmean': 'pottmp_wmean',
                'sltfl': 'sltfl', 'psl': 'msl', 'mlotst':'dbss_obml', 'sos':'salt', 'tauu': 'tau_x', 'tauv': 'tau_y', 'mld_diff': 'mld_diff',
            }
            in_channels = [godas_channel_dict.get(item.rstrip('_m')) + ('_m' if item.endswith('_m') else '') for item in in_channels]
            
        elif 'ORAS5' in in_models:
            file_path = "/mnt/disk1/ctefnet_data/ReanalysisVar/ORAS5/"
            full_time = pd.date_range(start='19580101', end='20231201', freq="MS")
            channel_dict = {
                'tos': 'votemper_5', 'thetao_5': 'votemper_5', 'thetao_20': 'votemper_20', 'thetao_40': 'votemper_40',
                'thetao_60': 'votemper_60', 'thetao_90': 'votemper_90', 'thetao_120': 'votemper_120',
                'thetao_150': 'votemper_150',
                'uo_5': 'vozocrtx_5', 'vo_5': 'vomecrty_5', 'wo_10': 'dzdt_10',
                'tauuo': 'sozotaux', 'tauvo': 'sometauy', 'zos': 'sshg', 'vor': 'vor',
                'hfds': 'thflx', 'thetao_wmean': 'votemper_wmean',
                'sltfl': 'sltfl', 'psl': 'msl', 'mlotst': 'somxl030', 'sos': 'sosaline', 'tauu': 'tau_x', 'tauv': 'tau_y','mld_diff': 'mld_diff'
            } # psl from ERA5
            in_channels = [channel_dict.get(item.rstrip('_m')) + ('_m' if item.endswith('_m') else '') for item in in_channels]
            
        else:
            file_path = "/mnt/disk1/ctefnet_data/CMIP6var/"

            # full_time = pd.date_range(start='18500101', end='20141201', freq="MS")
            full_time = pd.date_range(start='20150101', end='21001201', freq="MS")
        data = union_var(file_path, in_channels, in_models)
        index = load_index(file_path, out_channels, in_models)

        model_list = []
        index_list = []
        needed_time = (full_time.year >= time_range[0]) & (full_time.year <= time_range[1])

        print('preprocessing data ...')
        for i, model in tqdm(enumerate(data)):
            model_list.append(model[needed_time])
           
            if index:
                index_list.append(index[i][:, needed_time])
        del data
        del index
        print('done')

        self.data = model_list
        self.index = index_list
        self.num_model = len(self.data)
        self.num_mon = self.data[0].shape[0]
        self.model_len = self.num_mon - self.obs_time - self.pred_time

    def __len__(self):
        return self.num_model * self.model_len

    def __getitem__(self, idx):
        model = int(idx / self.model_len)
        month = int(idx % self.model_len)

        datax = torch.tensor(self.data[model][month: month + self.obs_time, :, self.input_region[0]:self.input_region[1], self.input_region[2]:self.input_region[3]], dtype=torch.float32)
        if self.transform is not None:
            datax = self.transform(datax)
        if self.output_type == 'index':
            datay = torch.tensor(self.data[model][month + self.obs_time: month + self.obs_time + self.pred_time, :,
                                 self.input_region[0]:self.input_region[1],
                                 self.input_region[2]:self.input_region[3]], dtype=torch.float32)
            if self.pred_type == 'series':
                index = torch.tensor(self.index[model][:, month + self.obs_time: month + self.obs_time + self.pred_time], dtype=torch.float32)
            else:
                index = torch.tensor(self.index[model][:, month + self.obs_time + self.pred_time, None], dtype=torch.float32)
            return datax, datay, index
        elif self.output_type == 'fig':
            datay = torch.tensor(self.data[model][month + self.obs_time: month + self.obs_time + self.pred_time, :, self.input_region[0]:self.input_region[1], self.input_region[2]:self.input_region[3]], dtype=torch.float32)
            return datax, datay
        

    def __init__(self, in_channels, out_channels, in_models, time_range, obs_time, pred_type, pred_time, input_region, target_region, output_type='index', transform=None):
        self.obs_time = obs_time
        self.pred_time = pred_time
        self.pred_type = pred_type
        self.output_type = output_type
        self.input_region = input_region
        self.target_region = target_region
        self.transform = transform
        if in_models == ['GODAS']:
            file_path = "/mnt/disk1/ctefnet_data/ReanalysisVar/GODAS/"
            full_time = pd.date_range(start='19800101', end='20231201', freq="MS")
            godas_channel_dict = {
                'tos': 'pottmp_5', 'thetao_5': 'pottmp_5', 'thetao_20': 'pottmp_20', 'thetao_40': 'pottmp_40',
                'thetao_60': 'pottmp_60', 'thetao_90': 'pottmp_90', 'thetao_120': 'pottmp_120',
                'thetao_150': 'pottmp_150',
                'uo_5': 'ucur_5', 'vo_5': 'vcur_5', 'wo_10': 'dzdt_10',
                'tauuo': 'uflx', 'tauvo': 'vflx', 'zos': 'sshg', 'vor': 'vor',
                'hfds': 'thflx', 'thetao_wmean': 'pottmp_wmean',
                'sltfl': 'sltfl', 'psl': 'msl', 'mlotst': 'dbss_obml', 'sos': 'salt', 'tauu': 'tau_x', 'tauv': 'tau_y'
            }
            in_channels = [godas_channel_dict.get(item.rstrip('_m')) + ('_m' if item.endswith('_m') else '') for item in
                           in_channels]
            # in_channels = [godas_channel_dict.get(item.rstrip('_mm')) + ('_mm' if item.endswith('_mm') else '') for item in in_channels]

        elif in_models == ['ORAS5']:
            file_path = "/mnt/disk1/ctefnet_data/ORAS5/"
            full_time = pd.date_range(start='19580101', end='20231201', freq="MS")
            channel_dict = {
                'tos': 'votemper_5', 'thetao_5': 'votemper_5', 'thetao_20': 'votemper_20', 'thetao_40': 'votemper_40',
                'thetao_60': 'votemper_60', 'thetao_90': 'votemper_90', 'thetao_120': 'votemper_120',
                'thetao_150': 'votemper_150',
                'uo_5': 'vozocrtx_5', 'vo_5': 'vomecrty_5', 'wo_10': 'dzdt_10',
                'tauuo': 'sozotaux', 'tauvo': 'sometauy', 'zos': 'sshg', 'vor': 'vor',
                'hfds': 'thflx', 'thetao_wmean': 'votemper_wmean',
                'sltfl': 'sltfl', 'psl': 'msl', 'mlotst': 'somxl030', 'sos': 'sosaline', 'tauu': 'tau_x', 'tauv': 'tau_y'
            }
            in_channels = [channel_dict.get(item.rstrip('_m')) + ('_m' if item.endswith('_m') else '') for item in
                           in_channels]
            # in_channels = [channel_dict.get(item.rstrip('_mm')) + ('_mm' if item.endswith('_mm') else '') for item in in_channels]



        else:
            file_path = "/mnt/disk1/ctefnet_data/CMIP6var/"
            full_time = pd.date_range(start='18500101', end='21001201', freq="MS")


        data = union_var(file_path, in_channels, in_models)
        index = load_index(file_path, out_channels, in_models)
        
        needed_time = (full_time.year >= time_range[0]) & (full_time.year <= time_range[1])      
        # print(needed_time)
        model_list = []
        index_list = []
        print('preprocessing data ...')
        for i, model in tqdm(enumerate(data)):
            model_list.append(model[needed_time])
            index_list.append(index[i][:, needed_time])
        del data
        del index
        print('done')

        self.data = model_list
        self.index = index_list
        self.num_model = len(self.data)
        self.num_mon = self.data[0].shape[0]
        self.model_len = self.num_mon - self.obs_time - self.pred_time + 1

    def __len__(self):
        return self.num_model * self.model_len

    def __getitem__(self, idx):
        model = int(idx / self.model_len)
        month = int(idx % self.model_len)

        datax = torch.tensor(
            self.data[model][month: month + self.obs_time, :, self.input_region[0]:self.input_region[1],
            self.input_region[2]:self.input_region[3]], dtype=torch.float32)
        if self.transform is not None:
            datax = self.transform(datax)
        if self.output_type == 'index':
            datay = torch.tensor(self.data[model][month + self.obs_time: month + self.obs_time + self.pred_time, :,
                                 self.input_region[0]:self.input_region[1],
                                 self.input_region[2]:self.input_region[3]], dtype=torch.float32)
            if self.pred_type == 'series':
                index = torch.tensor(
                    self.index[model][:, month + self.obs_time: month + self.obs_time + self.pred_time],
                    dtype=torch.float32)
            else:
                index = torch.tensor(self.index[model][:, month + self.obs_time + self.pred_time, None],
                                     dtype=torch.float32)
            return datax, datay, index
        elif self.output_type == 'fig':
            datay = torch.tensor(self.data[model][month + self.obs_time: month + self.obs_time + self.pred_time, :,
                                 self.input_region[0]:self.input_region[1], self.input_region[2]:self.input_region[3]],
                                 dtype=torch.float32)
            return datax, datay


if __name__=="__main__":
    from torch.utils.data import DataLoader


    ds = ENSODataloader(
        [ 'thetao_5','thetao_wmean', 'tauu', 'tauv', 'uo_5', 'vo_5', 'psl', 'mlotst','sos'],
        ['nino34'],
        ['ORAS5'],
        [1958, 1978],
        6,
        'series',
        20,
        [ 0, 120, 0, 180 ],
        [ 54, 64, 95, 120 ],
        output_type='index'
    )
    print(len(ds))
    #
    DL = DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    datax, datay, nino34 = next(iter(DL))
    # print(nino34)
    # print(datax.shape)
    # print(datay.shape)
    print(nino34.shape)
    # print(torch.any(torch.isnan(datax)))


