import torch
import copy
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import os.path as osp
from PIL import Image
import numpy as np
from utils.hparams import hparams
import torchvision.transforms as T
from models.positional_encoding import PositionalEncoder
from .module_util import make_layer, initialize_weights
from .commons import Mish, SinusoidalPosEmb, RRDB, Residual, Rezero, LinearAttention
from .commons import ResnetBlock, Upsample, Block, Downsample
import functools


class ConvBlock(nn.Module):
    """
    CONV Block to replace the MHSA layer in the Swin Transformer
    Window Self-attention
    This ConvBlock replaces the self-attention mechanism with

    1. Separable Convolution: depthwise and pointwise
    2. Batch Normalization
    3. RELU activation

    convolutions, followed by batch normalization and a GELU activation.
    """

    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        # Depthwise Convolution is applied to each input channel
        # independently, which captures spatial information.
        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=dim,
        )
        # Pointwise Convolution (1x1) is then applied to combine information
        # across different channels, which captures cross-channel dependencies.

        self.pointwise_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim)
        self.activation = nn.GELU()

    def forward(self, x):
        # Apply depthwise and pointwise convolutions
        x = self.conv(x)
        x = self.pointwise_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channel_size=64, kernel_size=3):
        """
        Args:
            channel_size : int, number of hidden channels
            kernel_size : int, shape of a 2D kernel
        """
        super(ResidualBlock, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_size,
                out_channels=channel_size,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=channel_size,
                out_channels=channel_size,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.PReLU(),
        )

    def forward(self, x):
        """
        Args:
            x : tensor (B, C, W, H), hidden state
        Returns:
            x + residual: tensor (B, C, W, H), new hidden state
        """
        residual = self.block(x)
        return x + residual


class HighResnetEncoder(nn.Module):
    def __init__(self, config):
        """
        Args:
            config : dict, configuration file
        """
        super(HighResnetEncoder, self).__init__()
        in_channels = config["in_channels"]
        num_layers = config["num_layers"]
        kernel_size = config["kernel_size"]
        channel_size = config["channel_size"]
        padding = kernel_size // 2

        self.init_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=channel_size,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.PReLU(),
        )

        res_layers = [
            ResidualBlock(channel_size, kernel_size) for _ in range(num_layers)
        ]
        self.res_layers = nn.Sequential(*res_layers)

        self.final = nn.Sequential(
            nn.Conv2d(
                in_channels=channel_size,
                out_channels=channel_size,
                kernel_size=kernel_size,
                padding=padding,
            )
        )

    def forward(self, x):
        """
        Encodes an input tensor x.
        Args:
            x : tensor (B, C_in, W, H), input images
        Returns:
            out: tensor (B, C, W, H), hidden states
        """
        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
        return x


class TALayer(nn.Module):
    def __init__(
        self,
        dim=64,
        num_heads=16,
        qkv_bias=True,
        qk_scale=None,
        drop=0.2,
        attn_drop=0.0,
        drop_path=0.0,
        proj_drop=0.0,
        T=1000,
        positional_encoding=True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        # self.d_model = d_model
        # @hubert from https://github.com/xianlin7/ConvFormer/blob/main/models/SETR.py#L73
        # self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, padding=1, bias=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.conv_block = ConvBlock(dim, kernel_size=3, stride=1, padding=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_spa_tmp = nn.Linear(2 * dim, dim)
        self.proj_drop_spa_tmp = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.drop_path = drop_path
        self.drop = drop

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.dim // num_heads, T=T, repeat=num_heads
            )
        else:
            self.positional_encoder = None

    def forward(self, x, batch_positions):
        """Forward function.

        Args:
            x: input features with shape of (B, T, N, C)
            mask: (0/-inf) mask with shape of (B, Wh*Ww, Wh*Ww) or None
        """
        # B, T, N, C = x.shape
        # print("Spatial-temporal Conv block input:", x.shape)

        if x.dim() == 5:
            B, T, C, H, W = x.shape  # torch.Size([4, 100, 64])
            N = H * W
            x = x.reshape(
                B, T, H * W, C
            )  # .permute(0, 3, 1, 2) # B, C, H, W :: torch.Size([4, 64, 40, 40])
        else:
            B, T, N, C = x.shape  # torch.Size([4, 6, 100, 64])

        H = W = int(math.sqrt(N))
        x_sp = x.view(B, T, H, W, C).permute(
            0, 1, 4, 2, 3
        )  # B, T, C, H, W :: torch.Size([36, 6, 16, 64])

        # Extract spatial information using convolutions
        x_sp = [
            self.conv_block(x_sp[:, i, :, :, :]) for i in range(T)
        ]  # input conv: B, C_in, H, W
        x_sp = torch.stack(x_sp, 1)  # expected output wieder: B,T,C_out,H,W
        x_sp = x_sp.view(B, T, C, N).permute(0, 1, 3, 2)  # B, T, N, C

        # Full computation of Spatio-Temporal attention at the same time
        # This is ok for lower-resolution images as the dimension
        # of the Q,K are of dim: N * T which could be heavy for large images

        # Extract temporal information using attetions
        x_t = x.permute(0, 2, 1, 3)  # from  B x T x N x C -->  B x N x T x C
        x_t = x_t.reshape(B * N, T, C)

        bp = (
            batch_positions.unsqueeze(-1)
            .repeat((1, 1, H))
            .unsqueeze(-1)
            .repeat((1, 1, 1, W))
        )  # BxTxHxW
        bp = bp.permute(0, 2, 3, 1).contiguous().view(B * H * W, T)

        # Add temporal positional encoding
        x_t = x_t + self.positional_encoder(bp)

        qkv_t = (
            self.qkv(x_t)
            .reshape(B * N, T, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]
        q_t = q_t * self.scale
        attn_t = q_t @ k_t.transpose(-2, -1)
        attn_t = self.softmax(attn_t)
        attn_t = self.attn_drop(attn_t)

        x_t = (attn_t @ v_t).transpose(1, 2).reshape(B * N, T, C)
        x_t = x_t.view(B, N, T, C).permute(0, 2, 1, 3)
        x = torch.cat((x_sp, x_t), dim=3)  # x should have size B x T x N x 2C

        x = self.proj_spa_tmp(x)  # size: B x T X N x C
        x = self.proj_drop_spa_tmp(x)  # size: B x T X N x C
        x = x.view(B, T, C, H, W)
        return x


class LTAE2d(nn.Module):
    def __init__(
        self,
        in_channels=64,
        n_head=16,
        d_k=4,
        mlp=[128, 64],
        dropout=0.2,
        d_model=128,
        T=1000,
        return_att=False,
        positional_encoding=True,
    ):
        """
        Lightweight Temporal Attention Encoder (L-TAE) for image time series.
        Attention-based sequence encoding that maps a sequence of images to a single feature map.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels (int): Number of channels of the input embeddings.
            n_head (int): Number of attention heads.
            d_k (int): Dimension of the key and query vectors.
            mlp (List[int]): Widths of the layers of the MLP that processes the concatenated outputs of the attention heads.
            dropout (float): dropout
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model.
            T (int): Period to use for the positional encoding.
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)
            positional_encoding (bool): If False, no positional encoding is used (default True).
        """
        super(LTAE2d, self).__init__()
        self.in_channels = in_channels
        self.mlp = copy.deepcopy(mlp)
        self.return_att = return_att
        self.n_head = n_head

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
        else:
            self.d_model = in_channels
            self.inconv = None
        assert self.mlp[0] == self.d_model

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.d_model // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )
        self.in_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=self.in_channels,
        )
        self.out_norm = nn.GroupNorm(
            num_groups=n_head,
            num_channels=mlp[-1],
        )

        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    nn.BatchNorm1d(self.mlp[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_positions=None, pad_mask=None, return_comp=False):
        sz_b, seq_len, d, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            )

        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, d)
        out = self.in_norm(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.inconv is not None:
            out = self.inconv(out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.positional_encoder is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # BxTxHxW
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)
            out = out + self.positional_encoder(bp)

        out, attn = self.attention_heads(out, pad_mask=pad_mask)

        out = (
            out.permute(1, 0, 2).contiguous().view(sz_b * h * w, -1)
        )  # Concatenate heads
        out = self.dropout(self.mlp(out))
        out = self.out_norm(out) if self.out_norm is not None else out
        out = out.view(sz_b, h, w, -1).permute(0, 3, 1, 2)

        attn = attn.view(self.n_head, sz_b, h, w, seq_len).permute(
            0, 1, 4, 2, 3
        )  # head x b x t x h x w

        if self.return_att:
            return out, attn
        else:
            return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, v, pad_mask=None, return_comp=False):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        q = torch.stack([self.Q for _ in range(sz_b)], dim=1).view(
            -1, d_k
        )  # (n*b) x d_k

        k = self.fc1_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        attn = attn.view(n_head, sz_b, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, sz_b, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature
        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        if return_comp:
            comp = attn
        # compat = attn
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        if return_comp:
            return output, attn, comp
        else:
            return output, attn


class Decoder(nn.Module):
    def __init__(self, config):
        """
        Args:
            config : dict, configuration file
        """

        super(Decoder, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=config["deconv"]["in_channels"],
                out_channels=config["deconv"]["out_channels"],
                kernel_size=config["deconv"]["kernel_size"],
                stride=config["deconv"]["stride"],
            ),
            nn.PReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=config["deconv"]["out_channels"],
                out_channels=config["deconv"]["out_channels"],
                kernel_size=config["deconv"]["kernel_size"],
                stride=config["deconv"]["stride"],
            ),
            nn.PReLU(),
        )

        self.final = nn.Conv2d(
            in_channels=config["final"]["in_channels"],
            out_channels=config["final"]["out_channels"],
            kernel_size=config["final"]["kernel_size"],
            padding=config["final"]["kernel_size"] // 2,
        )

    def forward(self, x):
        """
        Decodes a hidden state x.
        Args:
            x : tensor (B, C, W, H), hidden states
        Returns:
            out: tensor (B, C_out, 3*W, 3*H), fused hidden state
        """
        # torch.Size([4, 12, 64, 40, 40])
        deconv_outs = []
        for i in range(x.shape[1]):
            deconv_outs.append(self.deconv(x[:, i, :, :]))
        # Fusing the outputs of all timesteps
        x = torch.stack(deconv_outs, 1)  # torch.Size([4, 12, 64, 80, 80])
        # torch.Size([4, 64, 80, 80])

        deconv2_outs = []
        for i in range(x.shape[1]):
            deconv2_outs.append(self.deconv2(x[:, i, :, :]))
        # Fusing the outputs of all timesteps
        x = torch.stack(deconv2_outs, 1)  # torch.Size([4, 12, 64, 160, 160])

        x_final = []
        for i in range(x.shape[1]):
            x_final.append(self.final(x[:, i, :, :]))
        # Fusing the outputs of all timesteps
        x = torch.stack(x_final, 1)  # torch.Size([4, 12, 3, 160, 160])
        return x


class RRDB_MISR_Encoder(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super().__init__()
        hidden_size = hparams["hidden_size"]
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, lrs, norm=True):
        batch_size, seq_len, c_in, heigth, width = lrs.shape

        if norm:
            lrs = (lrs + 1) / 2
        lrs = lrs.view(batch_size * seq_len, c_in, heigth, width)

        fea_first = fea = self.conv_first(lrs)
        for l in self.RRDB_trunk:
            fea = l(fea)
        trunk = self.trunk_conv(fea)
        fea = fea_first + trunk
        return fea


class RRDB_MISR_Decoder(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super().__init__()
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if hparams["sr_scale"] == 8:
            self.upconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, fea, norm=True):
        fea = self.lrelu(
            self.upconv1(F.interpolate(fea, scale_factor=2, mode="nearest"))
        )
        fea = self.lrelu(
            self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
        )
        if hparams["sr_scale"] == 8:
            fea = self.lrelu(
                self.upconv3(F.interpolate(fea, scale_factor=2, mode="nearest"))
            )
        fea_hr = self.HRconv(fea)
        out = self.conv_last(self.lrelu(fea_hr))
        out = out.clamp(0, 1)
        if norm:
            out = out * 2 - 1
        return out


# Therefore, extending SRDiff to multiple images setting is a simple
# matter of switching RDDB to a MISR backbone or using HighRes-net encoders


class HighResLtaeNet(nn.Module):
    """
    The network architecture is composed of three main components: an encoder, a temporal encoder, and a decoder.
    The encoder is a high-resolution network that processes the input images and extracts high-level features.
    The temporal encoder is a lightweight temporal attention encoder that processes the high-level features extracted
      by the encoder and generates a single feature map.
    The decoder is a high-resolution network that processes the feature map generated by the temporal encoder and generates the super-resolved images.

    Given a set of images {x_1,...,x_T} representing a time-series, a refence image (x_ref) is computed as the median of the first 9 images.
    The reference image is then replicated T times and concatenated with the input images.
    A hidden state is then computed for each concatenation(x_t, x_ref) using the encoder.
    The fusion network then fuses the hidden states recursively, by halving by two the number of LR states
    at each fusion step.

    The decoder reconstructs the HR images from the fused hidden state with a deconvolution layer followed by a final layer

    """

    def __init__(self, config):
        super(HighResLtaeNet, self).__init__()
        self.encoder = HighResnetEncoder(config["network"]["encoder"])
        self.temporal_encoder = TALayer()
        self.decoder = Decoder(config["network"]["decoder"])

    def forward(self, lrs, dates, config, get_fea=False):
        B, _T, C, H, W = lrs.shape
        alphas = torch.tensor([0 for i in range(lrs.shape[1])])
        batch_size, seq_len, c_in, heigth, width = lrs.shape
        lrs = lrs.view(-1, seq_len, c_in, heigth, width)
        alphas = alphas.view(-1, seq_len, 1, 1, 1)

        if hparams["misr_ref_image"] == "median":
            refs, _ = torch.median(
                lrs[:, :9], 1, keepdim=True
            )  # reference image aka anchor, shared across multiple views
            refs = refs.repeat(1, seq_len, 1, 1, 1)
        if hparams["misr_ref_image"] == "closest":
            closest = torch.argmin(torch.abs(dates), 1)
            refs = lrs[0, closest[0], ...][None, None, ...]
            for e, c in enumerate(closest[1:]):
                refs = torch.cat([refs, lrs[e, c, ...][None, None, ...]])
            refs = refs.repeat(1, seq_len, 1, 1, 1)

        stacked_input = torch.cat([lrs, refs], 2)  # tensor (B, L, 2*C_in, W, H)

        stacked_input = stacked_input.view(
            batch_size * seq_len, 2 * c_in, width, heigth
        )
        fea = self.encoder(stacked_input)
        fea = fea.view(
            batch_size,
            seq_len,
            config["network"]["encoder"]["channel_size"],
            width,
            heigth,
        )
        # torch.Size([4, 12, 3, 40, 40])
        fea = self.temporal_encoder(fea, dates)
        # torch.Size([4, 12, 3, 40, 40])
        out = self.decoder(fea)
        # torch.Size([4, 12, 3, 160, 160]) --> torch.Size([4, 12, 3, 40, 40])

        cropping_ratio = int(out.shape[-1] / 4)
        transform = T.CenterCrop((cropping_ratio, cropping_ratio))
        out = transform(out)
        # Resizing the spatial dimensions (H, W)
        x_inter = F.interpolate(
            out.reshape(B * _T, C, H, W),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )
        # Reshape back to (B, _T, C, new_H, new_W)
        out = x_inter.view(B, _T, C, x_inter.shape[-2], x_inter.shape[-1])

        if get_fea:
            return out, fea
        else:
            return out


class RRDBLtaeNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, get_fea=False):
        super(RRDBLtaeNet, self).__init__()
        self.encoder = RRDB_MISR_Encoder(in_nc, out_nc, nf, nb, gc=32)
        self.temporal_encoder = TALayer(
            in_channels=32,
            n_head=16,
            d_k=4,
            mlp=[128, 32],
            dropout=0.2,
            d_model=128,
            T=1000,
            return_att=False,
            positional_encoding=True,
        )
        self.decoder = RRDB_MISR_Decoder(in_nc, out_nc, nf, nb, gc=32)

    def forward(self, lrs, dates, config, get_fea=False):
        batch_size, seq_len, c_in, heigth, width = lrs.shape
        out = self.encoder(lrs)
        out = out.view(
            batch_size,
            seq_len,
            config["network"]["encoder"]["channel_size"],
            width,
            heigth,
        )
        fea = self.temporal_encoder(out, dates)
        out = self.decoder(fea)

        if get_fea:
            return out, fea
        else:
            return out
