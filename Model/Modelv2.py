import torch
import torch.nn.functional as F
import torch.nn as nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
import torchvision.models as models
class InverseNet(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self,backbone):
        super().__init__()
        self.backbone = backbone
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        decoder_layer = nn.TransformerDecoderLayer(d_model=256, nhead=8,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        # hidden_dim = transformer.d_model
        self.input_proj = nn.Conv2d(512, 256, kernel_size=1)
        # self.query_embed = nn.Embedding(100, hidden_dim)
    def forward(self, samples1: NestedTensor, samples2: NestedTensor):
        if isinstance(samples1, (list, torch.Tensor)):
            samples1 = nested_tensor_from_tensor_list(samples1)
        if isinstance(samples2, (list, torch.Tensor)):
            samples2 = nested_tensor_from_tensor_list(samples2)
        # features1= self.backbone(samples1)
        features1, pos1 = self.backbone(samples1)
        features2, pos2 = self.backbone(samples2)
        srco1, mask1 = features1[-1].decompose()
        srco2, mask2 = features2[-1].decompose()
        src1 =  self.input_proj(srco1)
        src2 = self.input_proj(srco2)
        srcf1 = src1.view(src1.shape[0],src1.shape[1], src1.shape[2] * src1.shape[3]).transpose(1, 2)
        srcf2 = src2.view(src2.shape[0], src2.shape[1], src2.shape[2] * src2.shape[3]).transpose(1, 2)

        memory1 = self.transformer_encoder(srcf1)
        memory2 = self.transformer_encoder(srcf2)
        hs11 = self.transformer_decoder(srcf1,memory1)
        hs22 = self.transformer_decoder(srcf2,memory2)
        hs12 = self.transformer_decoder(srcf1,memory2)
        hs21 = self.transformer_decoder(srcf2,memory1)

        hs = torch.cat([hs11.unsqueeze(1),hs22.unsqueeze(1),hs12.unsqueeze(1),hs21.unsqueeze(1)],dim=1)
        # hs1 = self.transformer.Decoder(memory1,memory1,mask1,self.query_embed.weight, pos1[-1],bs1, c1, h1, w1)[0].transpose(1, 0)
        # hs2 = self.transformer.Decoder(memory2, memory2, mask2, self.query_embed.weight, pos2[-1], bs2, c2, h2, w2)[0].transpose(1, 0)
        # hs3 = self.transformer.Decoder(memory1,memory1,mask1,self.query_embed.weight, pos1[-1],bs1, c1, h1, w1)[0].transpose(1, 0)
        # hs4 = self.transformer.Decoder(memory2, memory2, mask2, self.query_embed.weight, pos2[-1], bs2, c2, h2, w2)[0].transpose(1, 0)
        return hs

class InverseTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50 = build_backbone()
        self.inversenet = InverseNet(resnet50)

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            # nn.Sigmoid()
        )

        # self.seq = nn.Sequential(
        #     nn.Conv2d(256, 128, 3, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, 5, 1),
        #     nn.ReLU(),
        # )
        self.featureconv = nn.Conv2d(4,1,1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x,y):
        #x shape batch size, 3, 600, 300
        #y shape batch size, 3, 600, 300
        x_inv = 255.0-x
        y_inv = 255.0 - y
        hsxy = self.featureconv(self.inversenet(x, y))
        featuresxy = hsxy.transpose(1, 3)
        feature2xy = self.gap(featuresxy).squeeze().squeeze()
        feature3xy = self.fc(feature2xy)

        hsxy_inv = self.featureconv(self.inversenet(x, y_inv))
        featuresxy_inv = hsxy_inv.transpose(1, 3)
        feature2xy_inv = self.gap(featuresxy_inv).squeeze().squeeze()
        feature3xy_inv = self.fc(feature2xy_inv)

        hsx_invy = self.featureconv(self.inversenet(x_inv, y))
        featuresx_invy = hsx_invy.transpose(1, 3)
        feature2x_invy = self.gap(featuresx_invy).squeeze().squeeze()
        feature3x_invy = self.fc(feature2x_invy)

        return feature3xy,feature3xy_inv,feature3x_invy
