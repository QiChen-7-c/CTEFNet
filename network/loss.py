import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(cfg):
    model_params = cfg.get('model')
    data_params = cfg.get('data')
    optim_params = cfg.get('optimizer')
    if model_params.get('name') in ['Geoformer', 'SimVP', 'SimVPV2', 'SimVPIT', 'DiffCast', 'SwinTransformer3D']:
        return FigLoss(
            optim_params.get('fig').get('loss_alpha'),
            optim_params.get('fig').get('loss_beta'),
            [
                data_params.get('target_region')[0] - data_params.get('input_region')[0],
                data_params.get('target_region')[1] - data_params.get('input_region')[0]+1,
                data_params.get('target_region')[2] - data_params.get('input_region')[2],
                data_params.get('target_region')[3] - data_params.get('input_region')[2]+1
            ],
        )
    elif model_params.get('name') in ['-']:
        return TokenLoss(
            optim_params.get('loss_alpha'),
            optim_params.get('loss_beta'),
        )
    else:
        return IndexLoss()
        # return MixLoss(
        #     optim_params.get('loss').get('lambda1'),
        #     optim_params.get('loss').get('lambda2'),
        #     optim_params.get('loss').get('lambda3'),
        #     data_params.get('obs_time')
        # )
        # if data_params.get('pred_type') == 'series':
        #     return IndexLoss()
        # else:
        #     return nn.MSELoss()


class IndexLoss(nn.Module):
    def __init__(self):
        super(IndexLoss, self).__init__()
        self.index_loss = nn.MSELoss()
        # ninoweight = (np.array([1.5] * 6 + [2] * 6 + [3] * 6 + [4] * 6 + [5] * 12) * np.log(np.arange(36) + 2))
        # self.ninoweight = torch.tensor(ninoweight, dtype=torch.float32, requires_grad=False, device='cuda')

    def forward(self, index_pred, index_true):
        # rmse = torch.sqrt(torch.mean((torch.norm(index_true, 1)+0.5)*(index_pred - index_true) ** 2, dim=(1,2)))  # [16, 2, 24] -> [24]
        # rmse = torch.sqrt(torch.mean((index_pred - index_true) ** 2 / (index_true ** 2 + 0.1), dim=0))
        rmse = torch.sqrt(torch.mean((index_pred - index_true) ** 2, dim=0))
        # rmse = self.index_loss(index_pred, index_true)
        # rmse = rmse * self.ninoweight[: rmse.shape[0]]
        return rmse.mean()
        # return self.index_loss(index_pred, index_true)


class MixLoss(nn.Module):
    def __init__(self, lambda_1, lambda_2, lambda_3, obs_time):
        super(MixLoss, self).__init__()
        self.lambda_1 = lambda_1  # day penalty
        self.lambda_2 = lambda_2  # regloss
        self.lambda_3 = lambda_3  # corrloss
        self.obs_time = obs_time

    def rmse(self, y_pred, y_true):
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
        # rmse = torch.sqrt(torch.mean((torch.norm(y_true, 2) + 0.1) * (y_pred - y_true) ** 2, dim=(1, 2)))
        return rmse.mean()

    def forward(self, index_pred, index_true):
        pred = index_pred[:, :, self.obs_time:]
        gtrue = index_true[:, :, self.obs_time:]
        regloss1 = self.rmse(pred, gtrue)

        if self.lambda_2 != 0:
            regloss2 = self.rmse(index_pred[:, :, :self.obs_time], index_true[:, :, :self.obs_time])
        else:
            regloss2 = 0

        if self.lambda_3 != 0:
            pred_ = torch.sub(pred, torch.mean(pred, dim=0))
            gtrue_ = torch.sub(gtrue, torch.mean(gtrue, dim=0))
            corr = 1 - F.cosine_similarity(pred_, gtrue_, dim=0)
            # corr = torch.maximum(corr, torch.zeros_like(corr, dtype=torch.float32))
            corr_loss = torch.mean(corr)
        else:
            corr_loss = 0

        return regloss1 * self.lambda_1 \
               + regloss2 * self.lambda_2 \
               + corr_loss * self.lambda_3



class TokenLoss(nn.Module):
    def __init__(self, lambda_1=1, lambda_2=1):
        super(TokenLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
    def forward(self, cls_pred, token_pred, true):
        # 16,24
        rmse1 = torch.sqrt(torch.mean((cls_pred - true) ** 2, dim=0)).sum()
        # 16,2340,24
        t = torch.sqrt(torch.mean((token_pred - true[:,None,:]) ** 2, dim=0))
        rmse2 = torch.minimum(t, 0.5*torch.ones_like(t, dtype=torch.float32)).mean(dim=0).sum()
        return self.lambda_1*rmse1 + self.lambda_2*rmse2


class FigLoss(nn.Module):
    def __init__(self, lambda_1, lambda_2, target_region):
        super(FigLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.target_region = target_region
        self.mse_loss1 = nn.MSELoss()
        self.mse_loss2 = nn.MSELoss()

    # def loss_var(self, y_pred, y_true):
    #     rmse = torch.mean((y_pred - y_true) ** 2, dim=[3, 4])
    #     rmse = rmse.sqrt().mean(dim=0)
    #     rmse = torch.sum(rmse, dim=[0, 1])
    #     return rmse
    #
    # def loss_nino(self, y_pred, y_true):  # MSELoss
    #     rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
    #     return rmse.sum()

    def forward(self, x, y):
        self.var_loss = self.mse_loss1(x, y)

        xi = x[:, :, 0, self.target_region[0]:self.target_region[1], self.target_region[2]:self.target_region[3]].mean(dim=[2, 3])
        yi = y[:, :, 0, self.target_region[0]:self.target_region[1], self.target_region[2]:self.target_region[3]].mean(dim=[2, 3])
        # xi = x[:, :, 0, self.target_region[0] - self.input_region[0]: self.target_region[1] -self.input_region[0]+1,
        # self.target_region[2] - self.input_region[2]: self.target_region[3] -self.input_region[2]+1].mean(dim=[2, 3])
        # yi = x[:, :, 0, self.target_region[0] - self.input_region[0]: self.target_region[1] -self.input_region[0]+1,
        # self.target_region[2] - self.input_region[2]: self.target_region[3] -self.input_region[2]+1].mean(dim=[2, 3])
        self.index_loss = self.mse_loss2(xi, yi)

        return self.var_loss * self.lambda_1 \
               + self.index_loss * self.lambda_2
        # return  self.index_loss


if __name__ == '__main__':
    criterion = MixLoss(1, 1, 1, 12)
    pred = torch.randn([8, 2, 36]).cuda()
    true = torch.randn([8, 2, 36]).cuda()
    loss = criterion(pred, true)
    # print(output)
    print(loss)
