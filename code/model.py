import timm
import torch
import torch, torch.nn as nn, torch.nn.functional as F
import torch
from torch import nn, einsum
import torchvision.models as models

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import torch.fft as fft
import cv2
# from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from transformers import VideoMAEConfig, VideoMAEForVideoClassification, VivitConfig, VivitForVideoClassification, AutoImageProcessor
#
# weights = Raft_Large_Weights.DEFAULT
# transforms = weights.transforms()
# https://pytorch.org/vision/stable/auto_examples/plot_optical_flow.html

class MAE(nn.Module):  # https://huggingface.co/docs/transformers/model_doc/videomae
    def __init__(self, enet_type=None, out_dim=9, valid_name='1'):
        super(MAE, self).__init__()
        # original config
        configuration = VideoMAEConfig(image_size = 224,
                                       patch_size = 16,
                                       num_channels = 3,
                                       num_frames = 16,
                                       tubelet_size = 2,
                                       hidden_size = 384,
                                       num_hidden_layers = 12,
                                       num_attention_heads = 12,
                                       intermediate_size = 1536,
                                       hidden_act = 'gelu',
                                       hidden_dropout_prob = 0.0,
                                       attention_probs_dropout_prob = 0.0,
                                       initializer_range = 0.02,
                                       layer_norm_eps = 1e-12,
                                       qkv_bias = True,
                                       use_mean_pooling = False,
                                       num_labels=out_dim,
                                       model_type='videomae')

        self.model = VideoMAEForVideoClassification(configuration)
        # self.model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-Kinetics",num_outputs=out_dim)
        # self.model = self.model(configuration)
        # self.image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        for param in self.model.parameters():
            param.requires_grad_(True)

    def forward(self, x):

        x = self.model(x)
        return x.logits


class VIVIT(nn.Module):  # https://huggingface.co/docs/transformers/model_doc/vivit
    def __init__(self, enet_type=None, out_dim=9, valid_name='1'):
        super(VIVIT, self).__init__()
        # model 줄인 config
        configuration = VivitConfig(image_size=224,
                                    num_frames=1,
                                    tubelet_size=[2, 16, 16],
                                    num_channels=3,
                                    hidden_size=384,
                                    num_hidden_layers=12,
                                    num_attention_heads=12,
                                    intermediate_size=1536,
                                    hidden_act='gelu_fast',
                                    hidden_dropout_prob=0.0,
                                    attention_probs_dropout_prob=0.0,
                                    initializer_range=0.02,
                                    layer_norm_eps=1e-06,
                                    qkv_bias=True,
                                    num_labels=out_dim)
        self.model = VivitForVideoClassification(configuration)
        # model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
    def forward(self, x):
        x = self.model(x)
        return x.logits

#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################

# Visual information(VI)
# class Efficient(nn.Module):
#     def __init__(self, enet_type=None, out_dim=15, valid_name='1'):
#         super(Efficient, self).__init__()
#         self.enet_name = 'tf_efficientnet_b0_ns'
#         self.enet = models.efficientnet_b0(img_size=224, pretrained=True)
#         # self.enet = models.efficientnet_b0(pretrained=True) # mi
#         # self.enet.classifier = nn.Sequential(nn.Linear(1280, out_dim))
#         self.enet.classifier = nn.Sequential()
#         # self.enet.classifier
#         # for param in self.enet.parameters():
#         #     param.requires_grad_(False)
#         self.fc = nn.Linear(1280, out_dim)
#         self.fc2 = nn.Linear(1280, 1)
#         self.sig = nn.Sigmoid()

#     def forward(self, x):
#         b, f, c, h, w = x.size()
#         x = x.view(-1, c, h, w)
#         # for i in range(f):
#         x = self.enet(x)
#         x = x.view(b, f, -1)
#         x = torch.mean(x, dim=1)
#         pred = self.fc(x)
#         event = self.fc2(x)
#         return pred, self.sig(event)

class Efficient(nn.Module):
    def __init__(self, enet_type=None, out_dim=4, valid_name='1'):
        super(Efficient, self).__init__()
        self.enet_name = 'tf_efficientnet_b0_ns'
        self.enet = models.efficientnet_b0(img_size=224, pretrained=True)
        self.enet.classifier = nn.Sequential()
        self.fc = nn.Linear(1280, out_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        
        # 입력 차원 확인
        if x.dim() == 4:  # 배치, 채널, 높이, 너비 (4차원)
            x = x.unsqueeze(1)  # 프레임 차원 추가: (b, 1, c, h, w)
            
        b, f, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.enet(x)
        x = x.view(b, f, -1)
        x = torch.mean(x, dim=1)
        pred = self.fc(x)
        return pred

#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            LinearWithRegularization(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            LinearWithRegularization(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MultiheadAttentionWithRegularization(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.5, l2_reg=0.01):
        super(MultiheadAttentionWithRegularization, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.l2_reg = l2_reg

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        output, attn_weights = self.attention(query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        # L2 regularization
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)

        # Add L2 regularization term to the attention output
        output = output + self.l2_reg * l2_loss

        return output, attn_weights

class LinearWithRegularization(nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_decay=0.01):
        super(LinearWithRegularization, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_decay = weight_decay  # L2 regularization strength

    def forward(self, x):
        # Apply L2 regularization to the linear layer weights
        l2_regularization = self.weight_decay * torch.sum(self.linear.weight ** 2) / 2
        output = self.linear(x) + l2_regularization
        return output


class Transformer(nn.Module):
    def __init__(self, dim=1280, depth=2, heads=8, dim_head=160, mlp_dim=2560, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = self.norm = nn.LayerNorm(dim)
        self.fc_q = LinearWithRegularization(dim, dim)
        self.fc_k = LinearWithRegularization(dim, dim)
        self.fc_v = LinearWithRegularization(dim, dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiheadAttentionWithRegularization(embed_dim=dim, num_heads=heads),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ]))

    def forward(self, query, key, value):
        base = key
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)
        for cross_attn, ff in self.layers:
            at, at_weight = cross_attn(query, key, value)
            x = self.norm(at) + base
            x = self.norm(ff(x)) + x
        return x


# class Cross_model(nn.Module):
#     def __init__(self, num_frames=10, out_dim=11, valid_name='1'):
#         super(Cross_model, self).__init__()

#         self.num_frames = num_frames
#         #######################################
#         self.eff = Efficient()
#         # self.eff.load_state_dict(torch.load(f'./weights/0826_fine_ShuffleVI_w10_best_fold_{valid_name}.pth_best_loss.pth'), strict=False)
#         self.eff.load_state_dict(torch.load(f'./weights/0826_fine_ShuffleVI_w10_best_fold_{valid_name}.pth_best_loss.pth'))
#         self.eff_feature_VI = nn.Sequential(list(self.eff.children())[0])
#         list(self.eff_feature_VI.children())[0].classifier = nn.Sequential()

#         for param in self.eff_feature_VI.parameters():
#             param.requires_grad_(False)

#         self.eff_MI = Efficient()
#         # self.eff_MI.load_state_dict(torch.load(f'./weights/0826_Shufflefine_MI_w30_best_fold_{valid_name}.pth_best_loss.pth'), strict=False)
#         self.eff_MI.load_state_dict(torch.load(f'./weights/0826_Shufflefine_MI_w30_best_fold_{valid_name}.pth_best_loss.pth'))
#         self.eff_feature_MI = nn.Sequential(list(self.eff_MI.children())[0])
#         list(self.eff_feature_MI.children())[0].classifier = nn.Sequential()

#         for param in self.eff_feature_MI.parameters():
#             param.requires_grad_(False)

# class Cross_model(nn.Module):
#     def __init__(self, num_frames=10, out_dim=11, valid_name='1'):
#         super(Cross_model, self).__init__()

#         self.num_frames = num_frames
#         #######################################
#         self.eff = Efficient()

#         # 가중치 로드 시 'module.' 접두사 제거 
#         eff_weights = torch.load(f'./weights/0828_AugVI_w30_best_fold_{valid_name}.pth.pth')  
#         # eff_weights = torch.load(f'./weights/1021_final_fine_VI_w30_best_fold_{valid_name}.pth.pth')  
#         # new_eff_weights = {k.replace('module.', ''): v for k, v in eff_weights.items()}
#         self.eff.load_state_dict(eff_weights)

#         self.eff_feature_VI = nn.Sequential(list(self.eff.children())[0])
#         list(self.eff_feature_VI.children())[0].classifier = nn.Sequential()

#         for param in self.eff_feature_VI.parameters():
#             param.requires_grad_(False)

#         self.eff_MI = Efficient()

#         # 가중치 로드 시 'module.' 접두사 제거 
#         eff_MI_weights = torch.load(f'./weights/0831_Aug2MI_w30_best_fold_{valid_name}.pth.pth') 
#         # eff_MI_weights = torch.load(f'./weights/1021_final_fine_MI_w30_best_fold_{valid_name}.pth.pth')  
#         # new_eff_MI_weights = {k.replace('module.', ''): v for k, v in eff_MI_weights.items()}
#         self.eff_MI.load_state_dict(eff_MI_weights)

#         self.eff_feature_MI = nn.Sequential(list(self.eff_MI.children())[0])
#         list(self.eff_feature_MI.children())[0].classifier = nn.Sequential()
        
#         max_len = 30
#         d_embed = 1280
#         encoding = torch.zeros(max_len, d_embed)
#         encoding.requires_grad = False
#         position = torch.arange(0, max_len).float().unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
#         encoding[:, 0::2] = torch.sin(position * div_term)
#         encoding[:, 1::2] = torch.cos(position * div_term)
#         self.encoding = encoding.unsqueeze(0)
        

#         self.VI_transformer = Transformer()
#         self.MI_transformer = Transformer()
#         self.classifier = nn.Sequential(LinearWithRegularization(1280*2, out_dim))

#     def forward(self, org_img, optical_img, device):
#         self.eff_feature_VI.eval()
#         self.eff_feature_MI.eval()

#         b, f, c, h, w = org_img.size()
#         org_img = org_img.view(-1, c, h, w)
#         org_img = self.eff_feature_VI(org_img)

#         org_img = org_img.view(b, f, -1) # batch, frame, feature
#         #######################################
#         #######################################
#         #######################################
#         b, f, c, h, w = optical_img.size()
#         optical_img = optical_img.view(-1, c, h, w)
#         optical_img = self.eff_feature_MI(optical_img)

#         optical_img = optical_img.view(b, f, -1) # batch, frame, feature
#         #######################################
#         #######################################
#         #######################################
#         org_img = org_img + self.encoding[:, :f, :].to(device) # batch, frame, feature
#         optical_img = optical_img + self.encoding[:, :f, :].to(device) # batch, frame, feature
#         #######################################
#         #######################################
#         VI_feature = self.VI_transformer(optical_img, org_img, org_img) # batch, frame, feature
#         MI_feature = self.MI_transformer(org_img, optical_img, optical_img) # batch, frame, feature
#         x_feature = torch.cat([VI_feature,MI_feature],dim=2) # # batch, frame, feature*2
#         x_feature = torch.mean(x_feature[:, :, :], axis=1) # batch, feature*2
#         x_feature = self.classifier(x_feature)
#         return x_feature


class Cross_model(nn.Module):
    def __init__(self, num_frames=10, out_dim=4, valid_name='1'):
        super(Cross_model, self).__init__()

        self.num_frames = num_frames
        #######################################
        self.eff = Efficient()

        # 가중치 로드 시 'module.' 접두사 제거 
        eff_weights = torch.load(f'./weights/1217_fine_VI_w10_best_fold_{valid_name}.pth.pth')  
        self.eff.load_state_dict(eff_weights)

        self.eff_feature_VI = nn.Sequential(list(self.eff.children())[0])
        list(self.eff_feature_VI.children())[0].classifier = nn.Sequential()

        for param in self.eff_feature_VI.parameters():
            param.requires_grad_(False)

        self.eff_MI = Efficient()

        # 가중치 로드 시 'module.' 접두사 제거 
        eff_MI_weights = torch.load(f'./weights/1217_fine_MI_w10_best_fold_{valid_name}.pth.pth') 
        self.eff_MI.load_state_dict(eff_MI_weights)

        self.eff_feature_MI = nn.Sequential(list(self.eff_MI.children())[0])
        list(self.eff_feature_MI.children())[0].classifier = nn.Sequential()
        
        max_len = 30
        d_embed = 1280
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0)
        

        self.VI_transformer = Transformer()
        self.MI_transformer = Transformer()
        self.classifier = nn.Sequential(LinearWithRegularization(1280*2, out_dim))

    def forward(self, org_img, optical_img, device):
        
        self.eff_feature_VI.eval()
        self.eff_feature_MI.eval()

        b, f, c, h, w = org_img.size()
        org_img = org_img.view(-1, c, h, w)
        org_img = self.eff_feature_VI(org_img)

        org_img = org_img.view(b, f, -1) # batch, frame, feature
        #######################################
        #######################################
        #######################################
        b, f, c, h, w = optical_img.size()
        optical_img = optical_img.view(-1, c, h, w)
        optical_img = self.eff_feature_MI(optical_img)

        optical_img = optical_img.view(b, f, -1) # batch, frame, feature
        #######################################
        #######################################
        #######################################
        org_img = org_img + self.encoding[:, :f, :].to(device) # batch, frame, feature
        optical_img = optical_img + self.encoding[:, :f, :].to(device) # batch, frame, feature
        #######################################
        #######################################
        VI_feature = self.VI_transformer(optical_img, org_img, org_img) # batch, frame, feature
        MI_feature = self.MI_transformer(org_img, optical_img, optical_img) # batch, frame, feature
        x_feature = torch.cat([VI_feature,MI_feature],dim=2) # # batch, frame, feature*2
        x_feature = torch.mean(x_feature[:, :, :], axis=1) # batch, feature*2
        x_feature = self.classifier(x_feature)
        return x_feature

#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
class Cross_sensor(nn.Module):
    def __init__(self, num_frames=10, out_dim=11, valid_name='1'):
        super(Cross_sensor, self).__init__()

        self.model = Cross_model()
        self.model.load_state_dict(torch.load(f'./weights/1109_cross_w30_best_fold_{valid_name}.pth'))
        self.model.classifier = nn.Sequential()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.sensor_fc = nn.Sequential(LinearWithRegularization(8, 300),
                                       nn.GELU(),
                                       LinearWithRegularization(300, 800),
                                       nn.GELU(),
                                       nn.Dropout(0.5),
                                       nn.BatchNorm1d(800),
                                       LinearWithRegularization(800, 1280)
                                       )
        self.transformer = Transformer()
        self.classifier = LinearWithRegularization(1280*3, out_dim)
    def forward(self, org_img, optical_img, device, sensor):
        b, f, c, h, w = org_img.size()
        self.model.eval()
        output = self.model(org_img, optical_img, device)
        # vi = output[:,:1280]
        # mi = output[:,1280:]
        b,t,data = sensor.size()

        sensor = sensor.view(-1, data)
        smart_feature = self.sensor_fc(sensor)
        smart_feature = smart_feature.view(b,t,-1)

        x_feature = self.transformer(smart_feature, smart_feature, smart_feature)  # batch, frame, feature
        # x_feature2 = self.transformer(smart_feature, mi, mi)  # batch, frame, feature
        # print(x_feature.shape)
        x_feature = torch.mean(x_feature[:, :, :], axis=1)
        x_feature = torch.cat([x_feature, output], dim=1)
        # x_feature = torch.mean(x_feature[:, :, :], axis=1)  # batch, feature*2
        x_feature = self.classifier(x_feature)

        return x_feature
    
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
    
class Concat_model(nn.Module):
    def __init__(self, num_frames=10, out_dim=11, valid_name='1'):
        super(Concat_model, self).__init__()

        self.num_frames = num_frames
        #######################################
        self.eff = Efficient()

        # 가중치 로드 시 'module.' 접두사 제거
        eff_weights = torch.load(f'./weights/0828_ShuffleVI_w30_best_fold_{valid_name}.pth.pth') # Shuffle
        # eff_weights = torch.load(f'./weights/0801_fine_VI_w30_best_fold_{valid_name}.pth_best_loss.pth') # No Shuffle
        # new_eff_weights = {k.replace('module.', ''): v for k, v in eff_weights.items()}
        self.eff.load_state_dict(eff_weights)

        self.eff_feature_VI = nn.Sequential(list(self.eff.children())[0])
        list(self.eff_feature_VI.children())[0].classifier = nn.Sequential()

        for param in self.eff_feature_VI.parameters():
            param.requires_grad_(False)

        self.eff_MI = Efficient()

        # 가중치 로드 시 'module.' 접두사 제거
        eff_MI_weights = torch.load(f'./weights/0828_ShuffleMI_w30_best_fold_{valid_name}.pth.pth') # Shuffle
        # eff_MI_weights = torch.load(f'./weights/0802_fine_MI_w30_best_fold_{valid_name}.pth_best_loss.pth') # No Shuffle
        # new_eff_MI_weights = {k.replace('module.', ''): v for k, v in eff_MI_weights.items()}
        self.eff_MI.load_state_dict(eff_MI_weights)

        self.eff_feature_MI = nn.Sequential(list(self.eff_MI.children())[0])
        list(self.eff_feature_MI.children())[0].classifier = nn.Sequential()
        
        max_len = 30
        d_embed = 1280
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0)

        self.classifier = nn.Sequential(LinearWithRegularization(1280*2, out_dim))

    def forward(self, org_img, optical_img, device):
        self.eff_feature_VI.eval()
        self.eff_feature_MI.eval()

        b, f, c, h, w = org_img.size()
        org_img = org_img.view(-1, c, h, w)
        org_img = self.eff_feature_VI(org_img)

        org_img = org_img.view(b, f, -1) # batch, frame, feature
        #######################################
        b, f, c, h, w = optical_img.size()
        optical_img = optical_img.view(-1, c, h, w)
        optical_img = self.eff_feature_MI(optical_img)

        optical_img = optical_img.view(b, f, -1) # batch, frame, feature
        #######################################
        org_img = org_img + self.encoding[:, :f, :].to(device) # batch, frame, feature
        optical_img = optical_img + self.encoding[:, :f, :].to(device) # batch, frame, feature
        #######################################
        x_feature = torch.cat([org_img, optical_img], dim=2) # batch, frame, feature*2
        x_feature = torch.mean(x_feature[:, :, :], axis=1) # batch, feature*2
        x_feature = self.classifier(x_feature)
        return x_feature
    
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################

class Cross_auto(nn.Module):
    def __init__(self, num_frames=30, out_dim=11, valid_name='1'):
        super(Cross_auto, self).__init__()

        self.model = Cross_model()
        self.model.load_state_dict(torch.load(f'./weights/1109_cross_w30_best_fold_{valid_name}.pth'))
        self.model.classifier = nn.Sequential()
        for param in self.model.parameters():
            param.requires_grad_(False)

        # self.sensor_fc = nn.Sequential(LinearWithRegularization(8, 300),
        #                                nn.GELU(),
        #                                LinearWithRegularization(300, 800),
        #                                nn.GELU(),
        #                                nn.Dropout(0.5),
        #                                nn.BatchNorm1d(800),
        #                                LinearWithRegularization(800, 1280)
        #                                )
        self.transformer = Transformer()
        self.classifier = nn.Sequential(LinearWithRegularization(1280*2, 1280),
                                        nn.GELU(),
                                        LinearWithRegularization(1280, out_dim))

        self.auto_fc = nn.Sequential(LinearWithRegularization(1280*2, 1280),
                                        nn.GELU(),
                                        LinearWithRegularization(1280, 1))
        self.sig = nn.Sigmoid()
    def forward(self, org_img, optical_img, device):
        b, f, c, h, w = org_img.size()
        self.model.eval()
        output = self.model(org_img, optical_img, device)
        # vi = output[:,:1280]
        # mi = output[:,1280:]
        # x_feature = torch.mean(x_feature[:, :, :], axis=1)  # batch, feature*2
        x_feature = self.classifier(output)
        x_auto = self.auto_fc(output)
        x_auto = self.sig(x_auto)

        return x_feature,x_auto


class Auto_classifer(nn.Module):
    def __init__(self, num_frames=30, out_dim=11, valid_name='1'):
        super(Auto_binary, self).__init__()

        self.model = Cross_model()
        self.model.load_state_dict(torch.load(f'./weights/1109_cross_w30_best_fold_{valid_name}.pth'))
        self.model.classifier = nn.Sequential()
        for param in self.model.parameters():
            param.requires_grad_(False)

        # self.sensor_fc = nn.Sequential(LinearWithRegularization(8, 300),
        #                                nn.GELU(),
        #                                LinearWithRegularization(300, 800),
        #                                nn.GELU(),
        #                                nn.Dropout(0.5),
        #                                nn.BatchNorm1d(800),
        #                                LinearWithRegularization(800, 1280)
        #                                )
        self.transformer = Transformer()
        self.classifier = nn.Sequential(LinearWithRegularization(1280*2, 1280),
                                        nn.GELU(),
                                        LinearWithRegularization(1280, out_dim))

        # self.auto_fc = nn.Sequential(LinearWithRegularization(1280*2, 1280),
        #                                 nn.GELU(),
        #                                 LinearWithRegularization(1280, 1))
        self.sig = nn.Sigmoid()
    def forward(self, org_img, optical_img, device):
        b, f, c, h, w = org_img.size()
        self.model.eval()
        output = self.model(org_img, optical_img, device)
        # vi = output[:,:1280]
        # mi = output[:,1280:]
        # x_feature = torch.mean(x_feature[:, :, :], axis=1)  # batch, feature*2
        x_feature = self.classifier(output)
        # x_auto = self.auto_fc(output)
        # x_auto = self.sig(x_auto)

        return x_feature, x_feature

class Auto_binary(nn.Module):
    def __init__(self, num_frames=30, out_dim=11, valid_name='1'):
        super(Auto_binary, self).__init__()

        self.model = Cross_model()
        self.model.load_state_dict(torch.load(f'./weights/1109_cross_w30_best_fold_{valid_name}.pth'))
        self.model.classifier = nn.Sequential()
        for param in self.model.parameters():
            param.requires_grad_(False)

        # self.sensor_fc = nn.Sequential(LinearWithRegularization(8, 300),
        #                                nn.GELU(),
        #                                LinearWithRegularization(300, 800),
        #                                nn.GELU(),
        #                                nn.Dropout(0.5),
        #                                nn.BatchNorm1d(800),
        #                                LinearWithRegularization(800, 1280)
        #                                )
        # self.transformer = Transformer()
        # self.classifier = nn.Sequential(LinearWithRegularization(1280*2, 1280),
        #                                 nn.GELU(),
        #                                 LinearWithRegularization(1280, out_dim))

        self.auto_fc = nn.Sequential(LinearWithRegularization(1280*2, 1280),
                                        nn.GELU(),
                                        LinearWithRegularization(1280, out_dim))
        self.sig = nn.Sigmoid()
    def forward(self, org_img, optical_img, device):
        b, f, c, h, w = org_img.size()
        self.model.eval()
        output = self.model(org_img, optical_img, device)
        # vi = output[:,:1280]
        # mi = output[:,1280:]
        # x_feature = torch.mean(x_feature[:, :, :], axis=1)  # batch, feature*2
        # x_feature = self.classifier(output)
        x_auto = self.auto_fc(output)
        x_auto = self.sig(x_auto)

        return x_auto, x_auto

#######################################
#######################################
#######################################
#######################################
#######################################
#######################################

class Self_VI(nn.Module):
    def __init__(self, num_frames=10, out_dim=11, valid_name='1'):
        super(Self_VI, self).__init__()

        self.num_frames = num_frames
        #######################################
        self.eff = Efficient()
        self.eff.load_state_dict(torch.load(f'./weights/1109_fine_VI_w10_best_fold_{valid_name}.pth'))
        # self.eff_feature = nn.Sequential(*list(self.eff.children())[0])
        self.eff_feature = nn.Sequential(list(self.eff.children())[0])
        list(self.eff_feature.children())[0].classifier = nn.Sequential()

        for param in self.eff_feature.parameters():
            param.requires_grad_(False)
        #######################################
        #######################################
        #######################################
        max_len = 10
        d_embed = 768
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0)
        #######################################
        #######################################
        #######################################
        self.projection_fc = nn.Linear(1280, 768)

        self.transformer = Transformer()
        self.classifier = nn.Linear(768, out_dim)
        # steering, speed, accel

    def forward(self, org_img, optical_img, device):
        self.eff_feature.eval()
        b, f, c, h, w = org_img.size()
        org_img = org_img.view(-1, c, h, w)
        org_img = self.eff_feature(org_img)

        org_img = org_img.view(-1, 1280)
        org_img = self.projection_fc(org_img)
        org_img = org_img.view(b, f, -1)
        #######################################
        #######################################
        org_img = org_img + self.encoding[:, :f, :].to(device)
        # optical_img = optical_img + self.encoding[:, :frame, :].to(device)
        #######################################
        #######################################

        x_feature = self.transformer(org_img, org_img, org_img) # batch, frame, feature
        x_feature = torch.mean(x_feature[:, :, :], axis=1) # batch, feature
        x_feature = self.classifier(x_feature)
        return x_feature


class Self_MI(nn.Module):
    def __init__(self, num_frames=10, out_dim=11, valid_name='1'):
        super(Self_MI, self).__init__()

        self.num_frames = num_frames
        #######################################
        self.eff = Efficient()
        self.eff.load_state_dict(torch.load(f'./weights/1109_fine_MI_w10_best_fold_{valid_name}.pth'))
        # self.eff_feature = nn.Sequential(*list(self.eff.children())[0])
        self.eff_feature = nn.Sequential(list(self.eff.children())[0])
        list(self.eff_feature.children())[0].classifier = nn.Sequential()

        for param in self.eff_feature.parameters():
            param.requires_grad_(False)
        #######################################
        #######################################
        #######################################
        max_len = 10
        d_embed = 768
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0)
        #######################################
        #######################################
        #######################################
        self.projection_fc = nn.Linear(1280, 768)

        self.transformer = Transformer()
        self.classifier = nn.Linear(768, out_dim)
        # steering, speed, accel

    def forward(self, org_img, optical_img, device):
        self.eff_feature.eval()
        b, f, c, h, w = org_img.size()
        org_img = org_img.view(-1, c, h, w)
        org_img = self.eff_feature(org_img)

        org_img = org_img.view(-1, 1280)
        org_img = self.projection_fc(org_img)
        org_img = org_img.view(b, f, -1)
        #######################################
        #######################################
        org_img = org_img + self.encoding[:, :f, :].to(device)
        # optical_img = optical_img + self.encoding[:, :frame, :].to(device)
        #######################################
        #######################################

        x_feature = self.transformer(org_img, org_img, org_img) # batch, frame, feature
        x_feature = torch.mean(x_feature[:, :, :], axis=1) # batch, feature
        x_feature = self.classifier(x_feature)
        return x_feature



#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################