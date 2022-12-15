# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .transformer import build_transformer



class InverseNet(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(100, hidden_dim)
    def forward(self, samples1: NestedTensor, samples2: NestedTensor,samples3: NestedTensor, samples4: NestedTensor):
        if isinstance(samples1, (list, torch.Tensor)):
            samples1 = nested_tensor_from_tensor_list(samples1)
        if isinstance(samples2, (list, torch.Tensor)):
            samples2 = nested_tensor_from_tensor_list(samples2)
        if isinstance(samples3, (list, torch.Tensor)):
            samples3 = nested_tensor_from_tensor_list(samples3)
        if isinstance(samples4, (list, torch.Tensor)):
            samples4 = nested_tensor_from_tensor_list(samples4)


        features1, pos1 = self.backbone(samples1)
        features2, pos2 = self.backbone(samples2)
        features3, pos3 = self.backbone(samples3)
        features4, pos4 = self.backbone(samples4)
        src1, mask1 = features1[-1].decompose()
        src2, mask2 = features2[-1].decompose()
        src3, mask3 = features3[-1].decompose()
        src4, mask4 = features4[-1].decompose()
        assert mask1 is not None
        assert mask2 is not None
        assert mask3 is not None
        assert mask4 is not None
        hs1 = self.transformer(self.input_proj(src1), mask1, self.query_embed.weight, pos1[-1])[0].transpose(1, 0)
        hs2 = self.transformer(self.input_proj(src2), mask2, self.query_embed.weight, pos2[-1])[0].transpose(1, 0)
        hs3 = self.transformer(self.input_proj(src3), mask3, self.query_embed.weight, pos3[-1])[0].transpose(1, 0)
        hs4 = self.transformer(self.input_proj(src4), mask4, self.query_embed.weight, pos4[-1])[0].transpose(1, 0)

        # hs1 = self.transformer.Decoder(memory1,memory1,mask1,self.query_embed.weight, pos1[-1],bs1, c1, h1, w1)[0].transpose(1, 0)
        # hs2 = self.transformer.Decoder(memory2, memory2, mask2, self.query_embed.weight, pos2[-1], bs2, c2, h2, w2)[0].transpose(1, 0)
        # hs3 = self.transformer.Decoder(memory1,memory1,mask1,self.query_embed.weight, pos1[-1],bs1, c1, h1, w1)[0].transpose(1, 0)
        # hs4 = self.transformer.Decoder(memory2, memory2, mask2, self.query_embed.weight, pos2[-1], bs2, c2, h2, w2)[0].transpose(1, 0)
        return hs1,hs2,hs3,hs4

class InverseTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50 = build_backbone()
        transfor = build_transformer()
        self.inversenet = InverseNet(resnet50, transfor)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # self.seq = nn.Sequential(
        #     nn.Conv2d(256, 128, 3, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, 5, 1),
        #     nn.ReLU(),
        # )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x,y):
        #x shape batch size, 3, 600, 300
        #y shape batch size, 3, 600, 300
        x_inv = 255.0-x
        y_inv = 255.0 - y
        x_feature,y_feature,xi_feature, yi_feature = self.inversenet(x, y,x_inv, y_inv)
        features = torch.cat([x_feature,y_feature], dim=1).transpose(1, 3)
        feature2 = self.gap(features).squeeze().squeeze()
        feature3 = self.fc(feature2)
        xi_features = torch.cat([xi_feature, y_feature], dim=1).transpose(1, 3)
        xi_feature2 = self.gap(xi_features).squeeze().squeeze()
        xi_feature3 = self.fc(xi_feature2)

        yi_features = torch.cat([xi_feature, yi_feature], dim=1).transpose(1, 3)
        yi_feature2 = self.gap(yi_features).squeeze().squeeze()
        yi_feature3 = self.fc(yi_feature2)

        return feature3,xi_feature3,yi_feature3





