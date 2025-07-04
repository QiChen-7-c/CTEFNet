import os
import math
import numpy as np

import torch
import torch.nn as nn
from torchvision import models
from einops.layers.torch import Rearrange

from .l0_layer import L0Dense
# from .vit import ViT


class BasicBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=(3, 5, 5), stride=stride, padding=(1, 2, 2), bias=False)
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=(3, 5, 5), stride=1, padding=(1, 2, 2), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channel)
        self.downsample = nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=(1, 1, 1), bias=False)

# 定义正向传播过程
    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class BasicBlock2D(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, bias=False)

# 定义正向传播过程
    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


class CTEFNet(nn.Module):
    def __init__(self, input_size = (120, 180), in_channels=10, dim=512, head=4, depth = 6, dim_feedforward = 256,
                 dropout = 0., obs_time=12, pred_time=20, num_index=1):
        super().__init__()
        self.dim = dim
        self.head = head
        self.obs_time = obs_time
        self.pred_time = pred_time
        self.num_index = num_index

        self.input_reg = L0Dense(in_channels, input_size[0], input_size[1], temperature=0.5, local_rep=True)
        # self.input_reg = nn.Sequential()

        # features extractor
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=(3, 4, 8), padding="same"),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            BasicBlock3D(in_channel=64, out_channel=64, stride=1),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            BasicBlock3D(in_channel=64, out_channel=64, stride=1),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            BasicBlock3D(in_channel=64, out_channel=128, stride=1),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            BasicBlock3D(in_channel=128, out_channel=256, stride=1),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),

            nn.AdaptiveAvgPool3d((None, 1, 1)),
            Rearrange('b c n h w -> b n c h w'),
            nn.Flatten(2),
            nn.Linear(in_features=256, out_features=dim, bias=True)
        )

        # temporal embedding
        pe = torch.zeros(obs_time+pred_time, dim)
        temp_position = torch.arange(0, obs_time+pred_time).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(temp_position * div_term)
        pe[:, 1::2] = torch.cos(temp_position * div_term)
        self.time_embedding = nn.Parameter(pe[None, ], requires_grad=False)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(dim, head, dim_feedforward = dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)
        decoder_layer = nn.TransformerDecoderLayer(dim, head, dim_feedforward = dim_feedforward, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, depth)


        self.deocder_head = nn.Sequential(
            nn.Linear(dim, self.num_index),
        )
        self.encoder_head = nn.Sequential(
            nn.Linear(obs_time * dim, pred_time * num_index),
        )
        self.res = nn.Parameter(torch.rand(dim, dtype=torch.float32), requires_grad=True)
        self.res_norm = nn.LayerNorm(dim)
        self.norm = nn.LayerNorm(dim)

    def make_mask_matrix(self, sz, repeat=None):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        if repeat:
            mask = mask.unsqueeze(0).repeat([repeat, 1, 1])
        if torch.cuda.is_available():
            mask = mask.cuda()
        return mask

    def to_embedding(self, x):
        (B, N, C, H, W) = x.shape
        x = x.view(B * N, C, H, W)
        x = self.input_reg(x)
        x = x.view(B, N, C, H, W).permute(0, 2, 1, 3, 4)
        x = self.conv(x)
        # x = x.view(B, N, self.dim)
        return x

    def embedding_to_index(self, x):
        (B, N, dims) = x.shape
        x.view(B * N, dims)
        index = self.deocder_head(x)
        index = index.view(B, N, self.num_index)
        index = index.transpose(2, 1)
        return index

    def encode(self, src_embedding, mask=None):
        src_embedding = self.norm(src_embedding) + self.time_embedding[:, :self.obs_time, :]
        return self.encoder(src_embedding, mask)

    def decode(self, tgt_embedding, src_embedding, tgt_mask=None, src_mask=None):
        tgt_embedding = self.norm(tgt_embedding) + self.time_embedding[:, self.obs_time:self.obs_time + tgt_embedding.size(1), :]
        return self.decoder(tgt = tgt_embedding, memory = src_embedding, tgt_mask = tgt_mask, memory_mask = src_mask)

    def transformer_forward(self, predictor, predictand, in_mask=None, enout_mask=None, sv_ratio=0):
        en_out = self.encode(predictor, in_mask)
        
        if predictand is not None:
            with torch.no_grad():
                connect_inout = torch.cat(
                    [predictor[:, -1:], predictand[:, :-1]], dim=1
                )
                out_mask = self.make_mask_matrix(connect_inout.size(1))
                outvar_pred = self.decode(
                    connect_inout,
                    en_out,
                    out_mask,
                    enout_mask,
                )
            if sv_ratio > 1e-7:
                supervise_mask = torch.bernoulli(
                    sv_ratio * torch.ones([predictand.size(0), predictand.size(1) - 1, 1])
                ).cuda()
            else:
                supervise_mask = 0
            predictand = (
                    supervise_mask * predictand[:, :-1]
                    + (1 - supervise_mask) * outvar_pred[:, :-1]
            )
            predictand = torch.cat([predictor[:, -1:], predictand], dim=1)

            # predicting
            outvar_pred = self.decode(
                predictand,
                en_out,
                out_mask,
                enout_mask,
            )
        else:
            assert predictand is None
            predictand = predictor[:, -1:]
            for t in range(self.pred_time):
                # out_mask = self.make_mask_matrix(predictand.size(1))
                out_mask = None
                outvar_pred = self.decode(
                    predictand,
                    en_out,
                    out_mask,
                    enout_mask,
                )
                predictand = torch.cat([predictand, outvar_pred[:, -1:]], dim=1)
        return en_out, outvar_pred

    def forward(self, src, tgt, sv_ratio=1):
        src_embedding = self.to_embedding(src)
        if tgt is not None:
            with torch.no_grad():
                tgt_embedding = self.to_embedding(tgt)
        else:
            tgt_embedding = None

        en_out, de_out = self.transformer_forward(src_embedding, None, sv_ratio=sv_ratio)


        skip = self.encoder_head(self.res_norm(en_out + src_embedding * self.res).flatten(1))
        skip = skip.view(-1, self.num_index, self.pred_time)

        out = self.embedding_to_index(de_out)
        out = out + skip

        return out, tgt_embedding, de_out


if __name__ == '__main__':
    import numpy as np

    model = CTEFNet(input_size=(120, 180), dim=256, head=4, in_channels=8, obs_time=24, pred_time=20, dim_feedforward = 512, num_index=1).cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    model.eval()
    src = torch.rand([8, 24, 8, 120, 180],dtype=torch.float32).cuda()
    tgt = torch.rand([8, 20, 8, 120, 180], dtype=torch.float32).cuda()
    out, tgt_embedding, de_out = model(src, tgt, 1)
    # print(out)
    print(out.shape, tgt_embedding.shape, de_out.shape)
