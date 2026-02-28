import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from models.encoders.t_convformer import TConvFormer
from models.decoders.unet_decoder import UNetDecoder

# II.TConvFormer like Swin Architecture (Encoder + Decoder)
# An encoder is implemented here;
# 1. ConvFormerSits(For timeseries)
# A decoder is implemented here;
# 1. UPerHead
# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
# Description: It uses shifted window approach for computing self-attention
# Adapated from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
# Paper associated to it https://ieeexplore.ieee.org/document/9710580


class SITSSegmenter(nn.Module):
    def __init__(
        self,
        config,
        img_size,
        in_chans,
        embed_dim,
        uper_head_dim,
        depths,
        num_heads,
        mlp_ratio,
        num_classes,
        nbts,
        pool_scales,
        spa_temp_att,
        conv_spa_att,
        decoder_channels,
        window_size,
        d_model,
        dropout_ratio=0.1,
    ):
        super().__init__()
        self.backbone_dims = [embed_dim * 2**i for i in range(len(depths))]
        self.img_size = img_size
        self.num_classes = num_classes
        self.nbts = nbts
        self.d_model = d_model
        self.pool_scales = pool_scales
        self.pool_scales = pool_scales
        self.spa_temp_att = spa_temp_att
        self.conv_spa_att = conv_spa_att
        self.decoder_channels = decoder_channels
        self.config = config
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.dropout_ratio = dropout_ratio
        # self.partition_size = partition_size
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.sr_encoder = TConvFormer(
            input_size=(self.img_size, self.img_size),
            stem_channels=64,
            block_channels=[128, 256, 512],  # [64, 128, 256, 512]
            block_layers=[2, 2, 5],  # [2, 2, 5, 2]
            head_dim=32,
            stochastic_depth_prob=0.2,
            partition_size=8,
        )

        self.encoder_channels = [
            self.embed_dim,
            self.embed_dim * 2,
            self.embed_dim * 4,
            self.embed_dim * 8,
        ]

        self.sr_decoder = UNetDecoder(
            self.encoder_channels,
            self.decoder_channels,
            self.dropout_ratio,
            self.window_size,
            self.num_classes,
        )

    def forward(self, x, batch_positions=None):
        h, w = x.size()[-2:]
        red_temp_feats, enc_temp_feats = self.sr_encoder(x, batch_positions)
        # res0, res1, res2, res3 = self.sr_encoder(x)
        sits_logits, multi_lvls_cls = self.sr_decoder(red_temp_feats, h, w)
        # print(" ----------- sits_logits: ", sits_logits.shape)
        # sits_logits:  torch.Size([2, 13, 64, 64])

        
        # Unlike Kanyamahanga et al. 2025, we do not merge the features of different
        # epochs at the end of the decoder (enc_temp_feats). 
        return sits_logits, multi_lvls_cls, red_temp_feats, enc_temp_feats
