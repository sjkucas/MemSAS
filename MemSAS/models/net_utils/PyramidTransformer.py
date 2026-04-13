import torch
import torch.nn as nn
import torch.nn.functional as F
from models.net_utils.tgcn import ConvTemporalGraphical
import einops
from torch.nn.init import constant_
from torch import optim

from  models.net_utils.pos_embed import Pos_Embed

import copy
import numpy as np
import math

class Encoder_bound(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha,joint_num,segment_num, A):
        super(Encoder_bound, self).__init__()
        self.conv_1x1 = nn.Conv2d(input_dim, num_f_maps, 1)  # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', 0, joint_num,segment_num, A) for i in # 2**i
             range(num_layers)])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.conv_bound = nn.Conv1d(num_f_maps, 1, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate
        # =============================
        self.register_buffer(
            "prototypes",
            F.normalize(torch.randn(num_layers, num_classes * 2, 32, 64), dim=-1)
        )
        self.register_buffer(
            "prototypes_spa",
            F.normalize(torch.randn(num_layers, 8, 24, 64), dim=-1)
        )
        self.TCN_layer1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1, dilation=1),
            nn.GELU(), nn.BatchNorm1d(64, track_running_stats=False)
        )
        self.TCN_layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.GELU(), nn.BatchNorm1d(64, track_running_stats=False)
        )
        self.TCN_layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.GELU(), nn.BatchNorm1d(64, track_running_stats=False)
        )
        self.TCN_layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=8, dilation=8),
            nn.GELU(), nn.BatchNorm1d(64, track_running_stats=False)
        )
        self.curv = Curvature(10, 10)
        self.curv_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 4)
        )
        # =============================
    def forward(self, x, mask):
        '''
        :param x: (N, C, T,V)
        :param mask:
        :return:
        '''

        # if self.channel_masking_rate > 0:
        #     x = self.dropout(x)

        # feature = self.conv_1x1(x)
        # for layer in self.layers:
        #     feature = layer(feature, None, mask)
        #
        # N, C, T, V=feature.size()
        # feature = F.avg_pool2d(feature, kernel_size=(1, V))

        feature = self.conv_1x1(x)
        N, C, T, V = feature.size()
        feature1 = []
        for i, layer in enumerate(self.layers):
            # feature = layer(feature, None, mask, self.prototypes[i])
            feature = layer(feature, None, mask, self.prototypes[i], self.prototypes_spa[i])
            # feature = layer(feature, None, mask, None, None)
            feature1.append(F.avg_pool2d(feature, kernel_size=(1, V)).view(N, C, T))
        # N, C, T, V = feature.size()
        feature = F.avg_pool2d(feature, kernel_size=(1, V))
        # feature2 = feature1
        # ======================================
        feature1 = torch.stack(feature1, dim=0).mean(dim=0)
        pred_curv = self.curv(feature1)
        feature1_1 = self.TCN_layer1(feature1)
        feature1_2 = self.TCN_layer2(feature1)
        feature1_3 = self.TCN_layer3(feature1)
        feature1_4 = self.TCN_layer4(feature1)
        # feature1 = (feature1_1 + feature1_2 + feature1_3 + feature1_4) / 4.0
        curv = pred_curv.unsqueeze(-1)  # [B, T, 1]
        curv_weight = self.curv_mlp(curv)  # [B, T, 4]
        curv_weight = torch.softmax(curv_weight, dim=-1)
        feature1 = torch.stack(
            [feature1_1, feature1_2, feature1_3, feature1_4],
            dim=-1
        )
        curv_weight = curv_weight.unsqueeze(1)  # [B, 1, T, 4]
        feature1 = (feature1 * curv_weight).sum(dim=-1)  # [B, C, T]
        # feature2 = [x + feature1 for x in feature2]
        # ======================================
        # M pooling
        feature = feature.view(N, C, T) #（batch，channel，temporal）
        out = self.conv_out(feature) * mask[:, 0:1, :] #（batch，class，temporal）
        bound = self.conv_bound(feature1) * mask[:, 0:1, :]
        # bound = self.conv_bound(feature) * mask[:, 0:1, :]
        return out, bound, feature1
        # return out, bound, feature

class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha,joint_num,segment_num, A):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv2d(input_dim, num_f_maps, 1)  # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha, joint_num,segment_num, A) for i in # 2**i
             range(num_layers)])

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

        # self.prototypes = nn.Parameter(
        #     torch.randn(num_layers, num_classes * 3, 16, 32)
        # )
        # self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)
        #=============================
        self.register_buffer(
            "prototypes",
            F.normalize(torch.randn(num_layers, num_classes * 2, 16, 64), dim=-1)
        )
        self.register_buffer(
            "prototypes_spa",
            F.normalize(torch.randn(num_layers, 8, 16, 64), dim=-1)
        )
        self.TCN_layer1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1, dilation=1),
            nn.GELU(), nn.BatchNorm1d(64, track_running_stats=False)
            )
        self.TCN_layer2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.GELU(), nn.BatchNorm1d(64, track_running_stats=False)
        )
        self.TCN_layer3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.GELU(), nn.BatchNorm1d(64, track_running_stats=False)
        )
        self.TCN_layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=8, dilation=8),
            nn.GELU(), nn.BatchNorm1d(64, track_running_stats=False)
        )
        self.curv = Curvature(10, 10)
        self.curv_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 4)
        )
        #=============================
    def forward(self, x, mask):
        '''
        :param x: (N, C, T,V)
        :param mask:
        :return:
        '''

        # if self.channel_masking_rate > 0:
        #     x = self.dropout(x)

        feature = self.conv_1x1(x)
        N, C, T, V = feature.size()
        feature1 = []
        for i, layer in enumerate(self.layers):
            # feature = layer(feature, None, mask, self.prototypes[i])
            # feature = layer(feature, None, mask, None, self.prototypes_spa[i])
            feature = layer(feature, None, mask, None, None)
            feature1.append(F.avg_pool2d(feature, kernel_size=(1, V)).view(N, C, T))
        # N, C, T, V = feature.size()
        feature = F.avg_pool2d(feature, kernel_size=(1, V))
        # feature2 = feature1
        #======================================
        feature1 = torch.stack(feature1, dim=0).mean(dim=0)
        pred_curv = self.curv(feature1)
        feature1_1 = self.TCN_layer1(feature1)
        feature1_2 = self.TCN_layer2(feature1)
        feature1_3 = self.TCN_layer3(feature1)
        feature1_4 = self.TCN_layer4(feature1)
        # feature1 = (feature1_1 + feature1_2 + feature1_3 + feature1_4) / 4.0
        curv = pred_curv.unsqueeze(-1)  # [B, T, 1]
        curv_weight = self.curv_mlp(curv)  # [B, T, 4]
        curv_weight = torch.softmax(curv_weight, dim=-1)
        feature1 = torch.stack(
            [feature1_1, feature1_2, feature1_3, feature1_4],
            dim=-1
        )
        curv_weight = curv_weight.unsqueeze(1)  # [B, 1, T, 4]
        feature1 = (feature1 * curv_weight).sum(dim=-1)  # [B, C, T]
        # feature2 = [x + feature1 for x in feature2]
        #======================================
        # M pooling
        feature = feature.view(N, C, T) #（batch，channel，temporal）
        out = self.conv_out(feature) * mask[:, 0:1, :] #（batch，class，temporal）
        # return out, feature2
        return out, feature1
        # return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.Downlayers = nn.ModuleList((
            AttModule_Decoder(2 ** 1, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 0),
            AttModule_Decoder(2 ** 1, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 1),
            AttModule_Decoder(2 ** 1, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 2),
            AttModule_Decoder(2 ** 1, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 3),

        ))
        self.Midlayers = nn.ModuleList((
            AttModule_Decoder(2 ** 1, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 4),
            AttModule_Decoder(2 ** 1, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 4),
        ))
        self.Uplayers = nn.ModuleList((
            AttModule_Decoder(2 ** 3, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 3),
            AttModule_Decoder(2 ** 5, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 2),
            AttModule_Decoder(2 ** 7, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 1),
            AttModule_Decoder(2 ** 9, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha, 2 ** 0),
        ))
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):

        feature = self.conv_1x1(x)
        mask_copy = mask
        masks = [mask_copy]

        for i in range(4):
            mask_copy = F.max_pool1d(mask_copy, kernel_size=2)
            masks.append(mask_copy)

        for index, layer in enumerate(self.Downlayers):
            feature = layer(feature, fencoder, masks[index])
            feature = F.avg_pool1d(feature, kernel_size=2)

        for index, layer in enumerate(self.Midlayers):
            feature = layer(feature, fencoder, masks[4])

        for index, layer in enumerate(self.Uplayers):
            feature = nn.functional.interpolate(feature, size=masks[3-index].shape[2], mode='linear')
            feature = layer(feature, fencoder, masks[3-index])


        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha, joint_num,segment_num, A):
        super(AttModule, self).__init__()

        self.GCN_layer = ConvFeedForward(A, in_channels, out_channels, dilation)
        self.convForReshapeV = self.conv_1x1 = nn.Conv1d(out_channels * segment_num, out_channels, 1)

        self.Vatt_layer = Spatial_AttLayer(in_channels, out_channels, in_channels // r1,
                                         num_frames=1,
                                         num_joints= joint_num,
                                         num_heads=3,
                                         kernel_size= (1,1),
                                         use_pes=True,
                                         att_drop= 0)

        self.TCN_layer = nn.Sequential( nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(dilation, 0), dilation=(dilation, 1)),
                                    nn.GELU(), nn.BatchNorm2d(in_channels, track_running_stats=False)
                                    )

        self.convForReshapeT = self.conv_1x1 = nn.Conv1d(out_channels * joint_num, out_channels, 1)
        self.instance_norm = nn.BatchNorm1d(in_channels)
        self.instance_norm2d = nn.BatchNorm2d(in_channels)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, 1, 1, 1, 64, att_type=att_type,
                                  stage=stage)  # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.joint_num =joint_num
        self.alpha = alpha
        self.segment_num = segment_num
        self.A = A

        self.feed_forward1 = ConvFeedForward_withoutGCN(joint_num, in_channels * 2, out_channels)
        self.feed_forward2 = ConvFeedForward_withoutGCN(joint_num, in_channels*2, out_channels)
        self.instance_norm2d2 = nn.BatchNorm2d(64)
        #==============================
        self.conv = nn.Conv1d(64, 16, kernel_size=1)
        self.act = nn.Hardswish()
        self.conv_t = nn.Conv1d(16, 64, kernel_size=1)
        self.conv_v = nn.Conv1d(16, 64, kernel_size=1)
        self.unit_gcn = unit_gcn(in_channels, out_channels, A)
        self.gate = GateNetwork(64, 8, 2)
        self.memory = Memory()
        #=============================
    def update_prototype(self, out, best_group_idx, prototype, K=2):
        B, C, G, T = out.shape
        score = out.mean(dim=1)  # [B,G,T]
        score = torch.softmax(score, dim=-1)
        topk_idx = score.topk(K, dim=-1).indices
        out_perm = out.permute(0, 2, 3, 1)  # [B,G,T,C]
        topk_feat = torch.gather(
            out_perm,
            2,
            topk_idx.unsqueeze(-1).expand(-1, -1, -1, C)
        )  # [B,G,K,C]
        proto_score = prototype.mean(dim=-1)
        proto_score = torch.softmax(proto_score, dim=-1)
        _, replace_idx = proto_score.topk(K, dim=-1, largest=False)
        with torch.no_grad():
            for b in range(B):
                for g in range(G):
                    p = best_group_idx[b, g]
                    prototype[p, replace_idx[p]] = topk_feat[b, g]

    def STCA(self, x, S):
        B, C, T, V = x.shape
        x = x.reshape(B, C, S, T // S, V)  # [B,C,S,T/S,V]
        x = x.permute(0, 2, 1, 3, 4)  # [B,S,C,T/S,V]
        x = x.reshape(B * S, C, T // S, V)  # [B*S,C,T/S,V]
        x_v = x.mean(dim=2)   # [B,64,19]
        x_t = x.mean(dim=3)   # [B,64,94]
        x_cat = torch.cat([x_v, x_t], dim=-1)  # [B,64,113]
        x_out = self.conv(x_cat)               # [B,32,113]
        x_out = self.act(x_out)
        x_v, x_t = torch.split(x_out, [V, T//S], dim=2)
        # conv
        x_t = self.conv_t(x_t)  # [B,64,94]
        x_v = self.conv_v(x_v)  # [B,64,19]
        # sigmoid
        x_t = torch.sigmoid(x_t)
        x_v = torch.sigmoid(x_v)
        # reshape for attention
        x_t = x_t.unsqueeze(-1)  # [B,64,94,1]
        x_v = x_v.unsqueeze(-2)  # [B,64,1,19]
        out = x * x_t * x_v
        out_mean = out.mean(dim=2)  # [128,64,19]
        out_mean= out_mean.view(B, self.segment_num, 64, self.joint_num)
        out_mean = out_mean.permute(0, 2, 1, 3)  # [2,64,segment_num,19]
        return out_mean

    def forward(self, x, f, mask, prototype=None, prototype_spa=None):
        gcn_out, _ = self.GCN_layer(x, self.A)
        N, C, T, V = gcn_out.size()  # N：batch C: channel T:temporal V:joint
        out = self.STCA(gcn_out, self.segment_num)
        # out = F.avg_pool2d(gcn_out, kernel_size=(T // self.segment_num, 1))
        out = out.permute(0, 2, 1, 3).contiguous().view(N * self.segment_num, C, V)  # （N*64，C，V）
        out = out.unsqueeze(2)  # (N, C * 64, 1, V)
        out, _ = self.unit_gcn(out, self.A)
        # out = self.Vatt_layer(out)
        out = out.squeeze(2)

        if prototype_spa is not None:
            if self.training:
                gating_coeffs, mask1 = self.gate(out)
                mask1 = mask1.bool()
                for i in range(prototype_spa.shape[0]):
                    global_compensation, prototype_spa[i] = self.memory(out, prototype_spa[i], self.training,
                                                                        mask1[:, i])
                    out = out + gating_coeffs[:, i].view(-1, 1, 1) * global_compensation
            else:
                gating_coeffs, mask1 = self.gate(out)
                mask1 = mask1.bool()
                for i in range(prototype_spa.shape[0]):
                    global_compensation = self.memory(out, prototype_spa[i], False,
                                                                        mask1[:, i])
                    out = out + gating_coeffs[:, i].view(-1, 1, 1) * global_compensation
        out = out.view(N, self.segment_num, C, V).permute(0, 2, 1, 3).contiguous()  # （N*64，C，V）
        out = nn.functional.interpolate(out, size=(T, V), mode='bilinear')
        out = torch.concat((gcn_out, out), dim=1)
        out = self.feed_forward1(out)

        tcn_out = self.TCN_layer(out)
        out = tcn_out.permute(0, 1, 3, 2).contiguous()  # （N,C,V,T）
        out = out.view(N, V * C, T)


        out = self.convForReshapeT(out)

        if prototype is not None:
            out1, best_group_idx = self.att_layer(self.instance_norm(out), f, mask, prototype)
            out1 = self.alpha * out1
            out = out1 + out
        else: out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        if self.training and prototype is not None:
            self.update_prototype(out.reshape(2, 64, -1, 64), best_group_idx, prototype)

        out = self.conv_1x1(out)
        out = self.dropout(out)
        out = out.unsqueeze(-1).expand(-1,-1,-1,self.joint_num)
        out = self.instance_norm2d(out)
        out = torch.concat((tcn_out, out),dim=1)
        out = self.feed_forward2(out)
        out = x + out
        out = self.instance_norm2d2(out)
        return  out * (mask[:, 0:1, :].unsqueeze(-1))


class AttModule_Decoder(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha, downrate):
        super(AttModule_Decoder, self).__init__()
        self.feed_forward = ConvFeedForward_Decoder(dilation, in_channels, out_channels)
        self.instance_norm = nn.BatchNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type,
                                  stage=stage, downrate= downrate)  # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha

    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]



class ConvFeedForward(nn.Module):
    def __init__(self, A, in_channels, out_channels,dilation):
        super(ConvFeedForward, self).__init__()
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, 3)

        self.edge_importance = nn.Parameter(torch.ones(A.size()))



    def _reset_parameters(self):
        constant_(self.layer[0].bias, 0.)

    def forward(self, x, A):
        out, A = self.gcn(x,A * self.edge_importance) #gcn


        return out, A


class ConvFeedForward_withoutGCN(nn.Module):
    def __init__(self, joint_num, in_channels, out_channels):
        super(ConvFeedForward_withoutGCN, self).__init__()

        self.convRS = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), dilation=(1, 1)),
            nn.GELU(), nn.BatchNorm2d(out_channels, track_running_stats=False)

            )

    def _reset_parameters(self):
        constant_(self.layer[0].bias, 0.)

    def forward(self, x):
        out = self.convRS(x)  # gcn
        return out

class ConvFeedForward_Decoder(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward_Decoder, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.GELU()
        )

    def forward(self, x):
        return self.layer(x)


class Spatial_AttLayer(nn.Module):
    def __init__(self, in_channels, out_channels, qkv_dim,
                 num_frames, num_joints, num_heads,
                 kernel_size, use_pes=True, att_drop=0):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pes = use_pes
        pads = int((kernel_size[1] - 1) / 2)
        # padt = int((kernel_size[0] - 1) / 2)

        # Spatio-Temporal Tuples Attention
        if self.use_pes: self.pes = Pos_Embed(in_channels, num_frames, num_joints)
        self.to_qkvs = nn.Conv2d(in_channels, 2 * num_heads * qkv_dim, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
        self.att0s = nn.Parameter(torch.ones(1, num_heads, num_joints, num_joints) / num_joints, requires_grad=True)
        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_heads, out_channels, (1, kernel_size[1]), padding=(0, pads)),
            nn.BatchNorm2d(out_channels))
        self.ff_net = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))


        if in_channels != out_channels:
            self.ress = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            self.rest = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.ress = lambda x: x
            self.rest = lambda x: x

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(att_drop)

    def forward(self, x):

        N, C, T, V = x.size()
        xs = self.pes(x) + x if self.use_pes else x
        q, k = torch.chunk(self.to_qkvs(xs).view(N, 2 * self.num_heads, self.qkv_dim, T, V), 2, dim=1)
        attention = self.tan(torch.einsum('nhctu,nhctv->nhuv', [q, k]) / (self.qkv_dim * T)) * self.alphas
        attention = attention + self.att0s.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        xs = torch.einsum('nctu,nhuv->nhctv', [x, attention]).contiguous().view(N, self.num_heads * self.in_channels, T, V)
        x_ress = self.ress(x)
        xs = self.relu(self.out_nets(xs) + x_ress)

        return xs

class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, dilaion, stage, att_type, downrate=1):  # r1 = r2
        super(AttLayer, self).__init__()

        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)

        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.downrate = downrate
        self.dilaion = dilaion
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder', 'decoder']

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2, mask, prototype=None):
        # x1 from the current stage
        # x2 from the last stage

        query = self.query_conv(x1)

        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
            key = self.key_conv(x2)
        else:
            value = self.value_conv(x1)
            key = self.key_conv(x1)

        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == 'sliding_att':
            if self.stage == 'encoder':
                if prototype is not None:
                    return self._sliding_window_self_att(query, key, value, mask, prototype)
                else: return self._sliding_window_self_att(query, key, value, mask)
            elif self.stage == 'decoder':
                return self._sliding_window_self_att_cross(query, key, value, mask)


    def _sliding_window_self_att(self, q, k, v, mask, prototype=None):
        QB, QE, QS = q.size()
        KB, KE, KS = k.size()
        VB, VE, VS = v.size()
        # padding zeros for the last segment
        # we want our sequence be dividable by  self.dilaion, so we need QS % self.dilaion == 0, if it is not the case we will pad it so it become
        nb = QS // self.dilaion
        if QS % self.dilaion != 0:
            q = F.pad(q, pad=(0, self.dilaion - QS % self.dilaion), mode='constant', value=0)
            k = F.pad(k, pad=(0, self.dilaion - QS % self.dilaion), mode='constant', value=0)
            v = F.pad(v, pad=(0, self.dilaion - QS % self.dilaion), mode='constant', value=0)
            nb += 1

        padding_mask = torch.cat([torch.ones((QB, 1, QS)).to(q.device) * mask[:, 0:1, :],
                                  torch.zeros((QB, 1, self.dilaion * nb - QS)).to(q.device)],
                                 dim=-1)

        # sliding window approach, by splitting query_proj and key_proj into shape (QE, l) x (QE, 2l)
        # sliding window for query_proj: reshape
        q = q.reshape(QB, QE, nb, self.dilaion).permute(0, 2, 1, 3).reshape(QB, nb, QE,
                                                                            self.dilaion)

        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = F.pad(k, pad=(self.dilaion // 2, self.dilaion // 2), mode='constant',
                  value=0)
        v = F.pad(v, pad=(self.dilaion // 2, self.dilaion // 2), mode='constant', value=0)
        padding_mask = F.pad(padding_mask, pad=(self.dilaion // 2, self.dilaion // 2), mode='constant',
                             value=0)


        # 2. reshape key_proj of shape (QB*nb, QE, 2*self.dilaion)
        k = torch.cat([k[:, :, i * self.dilaion:(i + 1) * self.dilaion + (self.dilaion // 2) * 2].unsqueeze(1) for i in
                       range(nb)],
                      dim=1)
        v = torch.cat([v[:, :, i * self.dilaion:(i + 1) * self.dilaion + (self.dilaion // 2) * 2].unsqueeze(1) for i in
                       range(nb)], dim=1)

        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat(
            [padding_mask[:, :, i * self.dilaion:(i + 1) * self.dilaion + (self.dilaion // 2) * 2].unsqueeze(1) for i in
             range(nb)], dim=1)  # （batch，1，temporal）变为（batch,temporal/dilaion，1，2×dilaion）

        # construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        window_mask = torch.zeros((1, self.dilaion, self.dilaion + 2 * (self.dilaion // 2))).to(
            q.device)  # window_mask （1,dilaion,dilaion×2）
        for i in range(self.dilaion):
            window_mask[:, :, i:i + self.dilaion] = 1

        final_mask = window_mask.unsqueeze(1).repeat(QB, nb, 1, 1) * padding_mask

        proj_query = q  # （batch，T/Dilaion，channel，Dilaion）
        proj_key = k  # （batch，T/Dilaion，channel，2×Dilaion）
        proj_val = v  # （batch，T/Dilaion，channel，2×Dilaion）
        padding_mask = final_mask

        b, m, QE, l1 = proj_query.shape
        b, m, KE, l2 = proj_key.shape

        # print(proj_query.shape)
        # print(proj_key.shape)
        # print(prototype.shape)
        #=============================
        if prototype is not None:
            core_prior = prototype.mean(dim=1)
            core_prior = F.normalize(core_prior, dim=-1)
            query_mean = proj_query.mean(dim=-1)
            query_mean = F.normalize(query_mean, dim=-1)  # 单位化
            sim = torch.einsum('bqc,gc->bqg', query_mean, core_prior)


            best_group_idx = sim.argmax(dim=-1)
            selected_prototypes = []
            for b1 in range(b):
                groups = []
                for q in range(m):
                    idx = best_group_idx[b1, q]  # 最佳组
                    group_proto = prototype[idx]  # [16,32]
                    groups.append(group_proto)
                selected_prototypes.append(torch.stack(groups))  # [94,16,32]
            selected_prototypes = torch.stack(selected_prototypes)

            # print(selected_prototypes.shape)
            key1 = selected_prototypes.transpose(2, 3)
            value1 = selected_prototypes.transpose(2, 3)
            energy1 = torch.einsum('n b k i, n b k j -> n b i j', proj_query,
                                  key1)
            attention1 = energy1 / (np.sqrt(QE) * 1.0)
            attention1 = attention1
            attention1 = self.softmax(attention1)
            attention1 = attention1
            output1 = torch.einsum('n b i k, n b j k-> n b i j', value1,
                                  attention1)
            # print(output1.shape)
            bb, cc, ww, hh = output1.shape
            output1 = einops.rearrange(output1, 'b c h w -> (b c) h w')
            output1 = einops.rearrange(output1, '(b c) h w->b c h w', b=bb, c=cc)
            output1 = output1.reshape(QB, nb, -1, self.dilaion).permute(0, 2, 1, 3).reshape(QB, -1,
                                                                                          nb * self.dilaion)
            output1 = output1[:, :, 0:QS]
            return output1 * mask[:, 0:1, :], best_group_idx
        #=============================
        energy = torch.einsum('n b k i, n b k j -> n b i j', proj_query,
                              proj_key)
        attention = energy / (np.sqrt(QE) * 1.0)
        attention = attention + torch.log( padding_mask + 1e-6)
        attention = self.softmax(attention)
        attention = attention * padding_mask
        output = torch.einsum('n b i k, n b j k-> n b i j', proj_val,
                              attention)
        # print(output.shape)
        bb, cc, ww, hh = output.shape
        output = einops.rearrange(output, 'b c h w -> (b c) h w')
        output = self.conv_out(F.gelu(output))
        output = einops.rearrange(output, '(b c) h w->b c h w', b=bb, c=cc)

        output = output.reshape(QB, nb, -1, self.dilaion).permute(0, 2, 1, 3).reshape(QB, -1,
                                                                                      nb * self.dilaion)
        output = output[:, :, 0:QS]
        # if prototype is not None:
        #     output = 0.5 *(output + output1)
        #     return output * mask[:, 0:1, :], best_group_idx
        return output * mask[:, 0:1, :]

    def _sliding_window_self_att_cross(self, q, k, v, mask):
        QB, QE, QS = q.size()
        KB, KE, KS = k.size()
        VB, VE, VS = v.size()

        Q_dilaion = self.dilaion
        KV_dilaion = self.dilaion * self.downrate

        nb = QS // Q_dilaion
        if QS % Q_dilaion != 0:
            q = F.pad(q, pad=(0, Q_dilaion - QS % Q_dilaion), mode='constant', value=0)
            nb += 1
        if KS % KV_dilaion != 0:
            k = F.pad(k, pad=(0, KV_dilaion - KS % KV_dilaion), mode='constant', value=0)
        if VS % KV_dilaion != 0:
            v = F.pad(v, pad=(0, KV_dilaion - VS % KV_dilaion), mode='constant', value=0)


        padding_mask = torch.cat([torch.ones((QB, 1, QS)).to(q.device) * mask[:, 0:1, :],
                                  torch.zeros((QB, 1, Q_dilaion * nb - QS)).to(q.device)],
                                 dim=-1)

        q = q.reshape(QB, QE, nb, Q_dilaion).permute(0, 2, 1, 3).reshape(QB, nb, QE,
                                                                            Q_dilaion)

        k = F.pad(k, pad=(KV_dilaion // 2, KV_dilaion // 2), mode='constant',
                  value=0)
        v = F.pad(v, pad=(KV_dilaion // 2, KV_dilaion // 2), mode='constant', value=0)
        padding_mask = F.pad(padding_mask, pad=(Q_dilaion // 2, Q_dilaion // 2), mode='constant',
                             value=0)


        k = torch.cat([k[:, :, i * KV_dilaion:(i + 1) * KV_dilaion + (KV_dilaion // 2) * 2].unsqueeze(1) for i in
                       range(nb)],
                      dim=1)
        v = torch.cat([v[:, :, i * KV_dilaion:(i + 1) * KV_dilaion + (KV_dilaion // 2) * 2].unsqueeze(1) for i in
                       range(nb)], dim=1)

        # 3. construct window mask
        padding_mask = torch.cat(
            [padding_mask[:, :, i * Q_dilaion:(i + 1) * Q_dilaion].unsqueeze(1) for i in
             range(nb)], dim=1).permute(0, 1, 3, 2).contiguous()

        # construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        window_mask = torch.zeros((1, Q_dilaion, KV_dilaion + 2 * (KV_dilaion // 2))).to(
            q.device)  # window_mask （1,dilaion,dilaion×2）
        for i in range(self.dilaion):
            window_mask[:, :, i:i + KV_dilaion] = 1

        final_mask = window_mask.unsqueeze(1).repeat(QB, nb, 1,
                                                     1) * padding_mask

        proj_query = q
        proj_key = k
        proj_val = v
        padding_mask = final_mask

        b, m, QE, l1 = proj_query.shape
        b, m, KE, l2 = proj_key.shape

        energy = torch.einsum('n b k i, n b k j -> n b i j', proj_query,
                              proj_key)
        attention = energy / (np.sqrt(QE) * 1.0)
        attention = attention + torch.log(
            padding_mask + 1e-6)
        attention = self.softmax(attention)
        attention = attention * padding_mask
        output = torch.einsum('n b i k, n b j k-> n b i j', proj_val,
                              attention)

        bb, cc, ww, hh = output.shape
        output = einops.rearrange(output, 'b c h w -> (b c) h w')
        output = self.conv_out(F.gelu(output))
        output = einops.rearrange(output, '(b c) h w->b c h w', b=bb, c=cc)

        output = output.reshape(QB, nb, -1, self.dilaion).permute(0, 2, 1, 3).reshape(QB, -1,
                                                                                      nb * self.dilaion)
        output = output[:, :, 0:QS]
        return output * mask[:, 0:1, :]

class unit_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 ratio=0.125,
                 intra_act='softmax',
                 inter_act='tanh',
                 norm='BN',
                 act='ReLU'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        num_subsets = A.size(0)
        self.num_subsets = num_subsets
        self.ratio = ratio
        mid_channels = int(ratio * out_channels)
        self.mid_channels = mid_channels

        self.norm_cfg = norm if isinstance(norm, dict) else dict(type=norm)
        self.act_cfg = act if isinstance(act, dict) else dict(type=act)
        self.act = nn.ReLU(inplace=True)
        self.intra_act = intra_act
        self.inter_act = inter_act

        self.A = nn.Parameter(A.clone())
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * num_subsets, 1),
            nn.BatchNorm2d(mid_channels * num_subsets), self.act)
        self.post = nn.Conv2d(mid_channels * num_subsets, out_channels, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(-2)
        self.alpha = nn.Parameter(torch.zeros(self.num_subsets))
        self.beta = nn.Parameter(torch.zeros(self.num_subsets))
        self.conv1 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)
        self.conv2 = nn.Conv2d(in_channels, mid_channels * num_subsets, 1)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""

        n, c, t, v = x.shape
        res = self.down(x)
        # K V V
        A = self.A
        A = A[None, :, None, None]

        """
        ***********************************
        *** Motion Topology Enhancement ***
        ***********************************
        """
        # The shape of pre_x is N, K, C, T, V
        pre_x = self.pre(x).reshape(n, self.num_subsets, self.mid_channels, t, v)
        x1, x2 = None, None
        # N C T V
        tmp_x = x
        # N K C T V
        x1 = self.conv1(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
        x2 = self.conv2(tmp_x).reshape(n, self.num_subsets, self.mid_channels, -1, v)
        # N K C 1 V
        x1 = x1.mean(dim=-2, keepdim=True)
        x2 = x2.mean(dim=-2, keepdim=True)
        graph_list = []
        # N K C 1 V V = N K C 1 V 1 - N K C 1 1 V
        diff = x1.unsqueeze(-1) - x2.unsqueeze(-2)
        # N K C 1 V V
        inter_graph = getattr(self, self.inter_act)(diff)
        inter_graph = inter_graph * self.alpha[0]
        # N K C 1 V V = N K C 1 V V + 1 K 1 1 V V
        A = inter_graph + A
        graph_list.append(inter_graph)
        # N K C 1 V * N K C 1 V = N K 1 1 V V
        intra_graph = torch.einsum('nkctv,nkctw->nktvw', x1, x2)[:, :, None]
        # N K 1 1 V V
        intra_graph = getattr(self, self.intra_act)(intra_graph)
        intra_graph = intra_graph * self.beta[0]
        # N K C 1 V V = N K 1 1 V V + N K C 1 V V
        A = intra_graph + A
        graph_list.append(intra_graph)
        A = A.squeeze(3)
        # N K C T V = N K C T V * N K C V V
        x = torch.einsum('nkctv,nkcvw->nkctw', pre_x, A).contiguous()
        # N K C T V -> N K*C T V
        x = x.reshape(n, -1, t, v)
        x = self.post(x)
        """
        ***********************************
        ***********************************
        ***********************************
        """

        get_gcl_graph = graph_list[0] + graph_list[1]
        # N K C 1 V V -> N K C V V
        get_gcl_graph = get_gcl_graph.squeeze(3)
        # N K C V V -> N K*C V V
        get_gcl_graph = get_gcl_graph.reshape(n, -1, v, v)

        return self.act(self.bn(x) + res), get_gcl_graph

class GateNetwork(nn.Module):
    def  __init__(self, input_size, num_experts, top_k):
        super(GateNetwork, self).__init__()
        # Global pooling layers to capture global context
        self.top_k = top_k
        # Two fully connected layers for scoring
        self.fc1 = nn.Linear(input_size, num_experts)
        self.relu1 = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        # Initialize fc1 weights to zero to stabilize training at start
        torch.nn.init.zeros_(self.fc1.weight)
    def forward(self, x):
        # Step 1. Global pooling → squeeze spatial dimension
        x = x.mean(dim=-1) + x.max(dim=-1).values
        # Step 2. Compute raw expert scores from fc1
        x = self.fc1(x)
        x= self.relu1(x)
        # Step 4. Add noise to scores and select Top-K experts
        topk_values, topk_indices = torch.topk(x, k=self.top_k, dim=1)
        # Step 5. Build mask for Top-K positions
        mask = torch.zeros_like(x).scatter_(dim=1, index=topk_indices, value=1.0)
        # Suppress all non-topK scores by setting them to -inf
        x[~mask.bool()] = float('-inf')
        # Step 6. Apply softmax across experts → gating distribution
        gating_coeffs = self.softmax(x)  # (batch_size, num_experts)
        # Each row sums to 1, only Top-K experts get non-zero weights
        return gating_coeffs,mask

class Memory(nn.Module):
    def __init__(self, ):
        super(Memory, self).__init__()
        # Number of top memory slots to attend in spatial refinement
        self.topk = 2

    def forward(self, feature, memory, train, mask):
        if train:
            global_compensation, updated_memory = self.Global_Pattern_Interaction_Stream(feature, memory, train, mask)
            return global_compensation, updated_memory
        else:
            global_compensation = self.Global_Pattern_Interaction_Stream(feature, memory, train, mask)
            return global_compensation
    def Global_Pattern_Interaction_Stream(self, image_feature, memory, train, mask):
        if train:
            I_G, global_compensation, score_memory, score_image = self.Global_Pattern_Adjustment(image_feature, memory)
            updated_memory, gathering_indices = self.Memory_Evolution(I_G, memory, score_memory, score_image, mask)
            return global_compensation, updated_memory
        else:
            I_G, global_compensation, score_memory, score_image = self.Global_Pattern_Adjustment(image_feature, memory)
            return global_compensation

    def Global_Pattern_Adjustment(self, feature, memory):
        attn = torch.softmax(feature.mean(dim=1), dim=-1)  # [B, J]
        feature_G = (feature * attn.unsqueeze(1)).mean(dim=-1)  # [B, C]
        score_memory, score_image = self.get_score(memory, feature_G)  # (B*h*w, M)
        memory_response = torch.matmul(score_image.detach(), memory)  # (B*h*w, d)
        memory_response = feature_G + memory_response  # (B*h*w, d)
        global_compensation = feature + memory_response.unsqueeze(-1)
        return feature_G, global_compensation, score_memory, score_image

    def get_score(self, memory, I_G):
        score = torch.matmul(I_G, torch.t(memory))  # (B, M)
        score_memory = F.softmax(score, dim=0)
        score_image = F.softmax(score, dim=1)
        return score_memory, score_image

    def Memory_Evolution(self, feature_G, memory, score_memory, score_image, mask):
        feature_G = feature_G[mask]
        score_image = score_image[mask]
        score_memory = score_memory[mask]
        if feature_G.shape[0] == 0:
            return memory, None  # 防止空 batch
        _, gathering_indices = torch.topk(score_image, 1, dim=1)  # (B*h*w, 1)
        weights = score_memory.gather(1, gathering_indices)  # (N, 1)
        update_vector = feature_G * weights  # (N, d)
        N, d = feature_G.shape
        # Aggregate into memory slots
        M = score_memory.size(1)
        output = torch.zeros((M, d), device=feature_G.device, dtype=feature_G.dtype)
        gathering_indices_exp = gathering_indices.expand(-1, d)  # (N, d)
        memory_increment = output.scatter_add_(0, gathering_indices_exp, update_vector)  # 聚合到 (M, d)
        updated_memory = F.normalize(memory_increment + memory, dim=1)  # (M, d)
        return updated_memory.detach(), gathering_indices

class Curvature(nn.Module):
    def __init__(self, q, w=10):
        super().__init__()
        self.q = q
        self.w = w

    def curvature_estimation(self, embs, q=10, device=0, w=10, normalization=None):
        embs = torch.tensor(embs, device=device)
        batch_size, emb_dim, seq_len = embs.shape

        # 根据选择的归一化方案进行处理
        if normalization is not None and normalization.lower() != 'none':
            # 计算归一化所需的范数
            if normalization.lower() == 'temporal':
                # 时序归一化：沿seq_len维度归一化
                norms = torch.norm(embs, p=2, dim=2, keepdim=True) + 1e-8
            elif normalization.lower() == 'channel':
                # 通道归一化：沿emb_dim维度归一化
                norms = torch.norm(embs, p=2, dim=1, keepdim=True) + 1e-8
            elif normalization.lower() == 'global':
                # 全局归一化：沿emb_dim和seq_len维度归一化
                norms = torch.norm(embs, p=2, dim=(1, 2), keepdim=True) + 1e-8
            else:
                raise ValueError(
                    f"不支持的归一化方案: {normalization}，可选方案为'none', 'temporal', 'channel', 'global'")

            # 应用L2归一化
            embs = embs / norms
        q = self.q
        w = self.w

        # 转置维度用于卷积操作
        # embs = embs.permute(0, 2, 1)  # [B, D, T]

        # 构建左右padding
        embs_pad_left = F.pad(embs, (q - 1, 0), mode='replicate')
        embs_pad_right = F.pad(embs, (0, q - 1), mode='replicate')

        # 构建差分卷积核
        kernel = torch.zeros(emb_dim, 1, q, device=embs.device)
        kernel[:, :, 0] = -1
        kernel[:, :, -1] = 1

        # 并行计算左右变化向量
        cv_left = F.conv1d(embs_pad_left, kernel, groups=emb_dim)
        cv_right = F.conv1d(embs_pad_right, kernel, groups=emb_dim)

        # 恢复原始维度
        cv_left = cv_left.permute(0, 2, 1)  # [B, T, D]
        cv_right = cv_right.permute(0, 2, 1)

        # 计算向量范数
        norm_left = torch.norm(cv_left, dim=2, keepdim=True)  # [B, T, 1]
        norm_right = torch.norm(cv_right, dim=2, keepdim=True)

        # 计算余弦相似度
        cos_sim = F.cosine_similarity(cv_left, cv_right, dim=2)
        cos_sim = torch.clamp(cos_sim, -1 + 1e-6, 1 - 1e-6)

        # 计算角度和曲率
        angle = torch.acos(cos_sim)
        curvature = angle / (norm_left.squeeze() + norm_right.squeeze() + 1e-6)

        # 移动平均平滑
        curvature = curvature.unsqueeze(1)  # [B, 1, T]
        curvature_pad = F.pad(curvature, (w - 1, w), mode='replicate')
        avg_kernel = torch.ones(1, 1, 2 * w, device=embs.device) / (2 * w)
        movavg = F.conv1d(curvature_pad, avg_kernel)

        # 反转并标准化到[0,1]
        movavg = movavg.squeeze(1)  # [B, T]
        movavg = movavg.max(dim=1, keepdim=True)[0] - movavg
        min_val = movavg.min(dim=1, keepdim=True)[0]
        max_val = movavg.max(dim=1, keepdim=True)[0]
        movavg = (movavg - min_val) / (max_val - min_val + 1e-6)

        curv = torch.squeeze(curvature)
        curv = torch.max(curv) - curv  # make CP higher than In-Segment
        curv = (curv - torch.min(curv)) / (torch.max(curv) - torch.min(curv))

        curv_reciprocal = ((norm_left.squeeze() + norm_right.squeeze()) / (angle + 1e-6))
        curv_reciprocal = (curv_reciprocal - torch.min(curv_reciprocal)) / (
                    torch.max(curv_reciprocal) - torch.min(curv_reciprocal))

        return curv, curv_reciprocal, movavg
    def forward(self, embs):
        # 计算预测曲率
        _, pred_curv, _ = self.curvature_estimation(embs)  # [B, T]
        return pred_curv