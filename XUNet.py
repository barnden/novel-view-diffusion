"""
"Novel View Synthesis with Diffusion Models" by Watson et al. (2022)

Preprint: https://arxiv.org/abs/2210.04628
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from Camera import *
from Embeddings import *


class GroupNorm(nn.Module):
    # GroupNorm over frames
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def __forward__(self, h):
        B, C, D, H, W = h.shape

        h = h.reshape(B * D, C, 1, H, W).squeeze()
        h = self.norm(h)
        h = h.reshape(B, C, D, H, W)

        return h


class FiLM(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.silu = nn.SiLU()
        self.dense = nn.Linear(features, 2 * features)

    def forward(self, x, emb):
        emb = self.silu(emb)
        emb = self.dense(emb)
        scale, shift = torch.split(emb, 2, dim=-1)

        x = x * (1.0 + scale) + shift

        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dropout, resample):
        super().__init__()

        self.norm1 = GroupNorm(num_channels=dim_in)
        self.conv1 = nn.Conv3d(dim_in, dim_out, kernel_size=(1, 3, 3), padding="same")
        self.norm2 = GroupNorm(num_channels=dim_out)
        self.film = FiLM(dim_out)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout3d(p=dropout)
        self.conv2 = nn.Conv3d(dim_out, dim_out, kernel_size=(1, 3, 3), padding="same")
        self.dense = nn.Linear(dim_in, dim_out)
        self.features = dim_out

        self.resample = (
            {
                # Watson et al. (2022): nearest_neighbor_upsample()
                "up": nn.Upsample(scale_factor=2, mode="nearest"),
                # Watson et al. (2022): avgpool_downsample()
                "down": nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            }[resample]
            if resample is not None
            else nn.Identity()
        )

        # From Watson et al.'s out_init_scale(), they use truncated normal with mean=0 and std=0
        nn.init.trunc_normal_(self.conv2.weight, mean=0.0, std=0.0)

    def forward(self, h_in, emb):
        C = h_in.shape[1]

        features = C if self.features is None else self.features

        h = self.norm1(h_in)
        h = self.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)

        h = self.film(h, emb)
        h = self.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if C != features:
            h_in = h_in.permute(0, 2, 3, 4, 1)
            h_in = self.dense(h_in)
            h_in = h_in.permute(0, 4, 1, 2, 3)

        h = (h + h_in) / math.sqrt(2)

        h = self.resample(h)

        return h


class AttnBlock(nn.Module):
    def __init__(self, features, attn_type=None, attn_heads=4):
        super().__init__()

        self.norm = GroupNorm(num_channels=features)
        self.attn_type = attn_type

        self.attn1 = nn.MultiheadAttention(features, attn_heads)
        self.attn2 = nn.MultiheadAttention(features, attn_heads)

        self.dense = nn.Linear(in_features=features, out_features=features)
        nn.init.trunc_normal_(self.dense.weight, mean=0.0, std=0.0)

    def forward(self, h_in):
        B, C, _, H, W = h_in.shape
        h = self.norm(h_in)

        h0 = h[:, :, 0].reshape(B, C, H * W)
        h1 = h[:, :, 1].reshape(B, C, H * W)

        if self.attn_type == "self":
            h0 = self.attn1(h0, h0, h0)
            h1 = self.attn2(h1, h1, h1)
        elif self.attn_type == "cross":
            h0 = self.attn1(h0, h1, h1)
            h1 = self.attn2(h1, h0, h0)
        else:
            raise NotImplementedError(self.attn_type)

        h = torch.stack((h0, h1), axis=2)

        h = h.permute(0, 2, 3, 4, 1)
        h = self.dense(h)
        h = h.permute(0, 4, 1, 2, 3)

        return (h + h_in) / math.sqrt(2)


class XUNetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, use_attn=False, attn_heads=4, dropout=0):
        super().__init__()

        self.use_attn = use_attn
        self.res = ResnetBlock(dim_in, dim_out, dropout)

        if use_attn:
            self.attn_self = AttnBlock(dim_in, "self", attn_heads)
            self.attn_cross = AttnBlock(dim_in, "cross", attn_heads)

    def forward(self, x, emb):
        h = self.res(x, emb)

        if self.use_attn:
            h = self.attn_self(h)
            h = self.attn_cross(h)

        return h


class ConditioningProcessor(nn.Module):
    def __init__(
        self,
        H,
        W,
        emb_ch,
        num_resolutions,
        use_pos_emb,
        use_ref_pose_emb,
    ):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(144, emb_ch), nn.SiLU(), nn.Linear(emb_ch, emb_ch)
        )

        self.convs = [
            nn.Conv3d(
                in_channels=1,
                out_channels=1,
                kernel_size=(1, 3, 3),
                stride=(1, 2**i, 2**i),
                padding=1,
            )
            for i in range(num_resolutions)
        ]
        # fmt: off
        magic = 16 * 9 # (15 - 0 + 1) (8 - 0 + 1) -> 16 * 9 (last dimension of pose_emb)
        magic_root = math.sqrt(magic)
        # Learnable positional embeddings
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_emb = torch.nn.Parameter(torch.empty(H, W, magic).normal_(std=1 / magic_root))
            self.pos_emb.requires_grad = True

        # Binary embedding to allow the model to distinguish frames
        self.use_ref_pose_emb = use_ref_pose_emb
        if use_ref_pose_emb:
            self.first_emb = torch.nn.Parameter(torch.empty(magic,).normal_(std=1 / magic_root))
            self.first_emb.requires_grad = True

            self.ref_pose_emb_other = torch.nn.Parameter(torch.empty(magic,).normal_(std=1 / magic_root))
            self.ref_pose_emb_other.requires_grad = True
        # fmt: on

    def forward(self, batch, cond_mask=None):
        # TODO: Classifier-free guidance over poses (cond_mask)
        H, W = batch["x"].shape[-2:]

        logsnr = torch.clip(batch["logsnr"], -20.0, 20.0)
        logsnr = 2.0 * torch.arctan(torch.exp(-logsnr / 2.0)) / torch.pi

        logsnr_emb = posenc_ddpm(logsnr, emb_ch=self.emb_ch, max_time=1.0)
        logsnr_emb = self.linear(logsnr_emb)

        pos, dir = Camera(H, W, batch["K"], batch["R"], batch["t"]).rays()

        pose_emb_pos = posenc_nerf(pos, min_deg=0, max_deg=15)
        pose_emb_dir = posenc_nerf(dir, min_deg=0, max_deg=8)
        pose_emb = torch.concat([pose_emb_pos, pose_emb_dir], axis=-1)

        if self.use_pos_emb:
            pose_emb += self.pos_emb[None, None]

        if self.use_ref_pose_emb:
            pose_emb += torch.concat([self.first_emb, self.other_emb])

        pose_embs = [conv(pose_emb) for conv in self.convs]

        return logsnr_emb, pose_embs


class XUNetLevel(nn.Module):
    def __init__(
        self, dim_in, dim_out, num_blocks=3, dropout=0, use_attn=True, attn_heads=4
    ):
        super().__init__()

        blocks = []

        for i in range(num_blocks):
            blocks.append(
                XUNetBlock(
                    dim_in=dim_in if i == 0 else dim_out,
                    dim_out=dim_out,
                    use_attn=use_attn,
                    attn_heads=attn_heads,
                    dropout=dropout,
                )
            )

        self.level = nn.Sequential(*blocks)

    def forward(self, h, emb):
        h = self.level(h, emb)

        return h


class XUNet(nn.Module):
    def __init__(
        self,
        ch=256,
        ch_mult=(1, 2, 2, 4),
        emb_ch=1024,
        num_res_blocks=3,
        attn_layers=(1, 2, 3),
        attn_heads=4,
        dropout=0.1,
        use_pos_emb=True,
        use_ref_pose_emb=True,
    ):
        super().__init__()

        self.ch_mult = ch_mult
        self.use_pos_emb = use_pos_emb
        self.use_ref_pose_emb = use_ref_pose_emb
        self.num_res_blocks = num_res_blocks
        self.attn_layers = attn_layers
        self.attn_heads = attn_heads
        self.dropout = dropout
        self.emb_ch = self.emb_ch

        self.processor = ConditioningProcessor(
            emb_ch=emb_ch,
            num_resolutions=len(ch_mult),
            use_pos_emb=use_pos_emb,
            use_ref_pose_emb=use_ref_pose_emb,
        )

        # Input (C, H, W) -> (ch, H, W)
        self.conv1 = nn.Conv3d(3, ch, kernel_size=(1, 3, 3), padding="same")

        # Encoder (ch, H, W) -> (ch * ch_mult[-1], 8, 8)
        self.encoder = []
        for i, level in enumerate(ch_mult):
            blocks = []

            use_attn = i in attn_layers
            features = ch * level

            for j in range(num_res_blocks):
                features_in = features if not i or j > 0 else ch * ch_mult[i - 1]
                blocks.append(XUNetBlock(features_in, features, use_attn, 4, dropout))

            blocks.append(ResnetBlock(features, features, dropout, resample="down"))

            self.encoder.append(blocks)

        # Bottleneck (ch * ch_mult[-1], 8, 8)
        self.bottleneck = XUNetBlock(ch * ch_mult[-1], ch * ch_mult[-2], use_attn=True, dropout=dropout)

        # Decoder (ch * ch_mult[-1], 8, 8) -> (ch, H, W)
        self.decoder = []
        for i, level in reversed(list(enumerate(ch_mult))):
            blocks = []

            use_attn = i in attn_layers
            features = ch * level

            for _ in range(num_res_blocks + 1):
                features_out = features if not i else ch * ch_mult[i - 1]
                blocks.append(
                    XUNetBlock(2 * features, features_out, use_attn, 4, dropout)
                )

            blocks.append(
                ResnetBlock(features_out, features_out, dropout, resample="up")
            )

            self.decoder.append(blocks)

        # Output (ch, H, W) -> (C, H, W)
        self.conv2 = nn.Conv3d(ch, 3, kernel_size=(1, 3, 3), padding="same")
        nn.init.trunc_normal_(self.conv2.weight, mean=0.0, std=0.0)

    def forward(self, batch):
        # TODO: Classifier-free guidance
        """
        batch: dict
            z: noisy input          (3, H, W)
            x: conditioning view    (3, H, W)
            logsnr: lambda          (1,)
            t: camera translations  2 x (3, 1)
            R: camera rotations     2 x (3, 3)
            K: camera intrinsic     (3, 3)
        """
        B, C, H, W = batch["x"].shape
        logsnr_emb, pose_embs = self.processor(batch)

        h = torch.stack([batch["x"], batch["z"]], axis=1)
        h = self.conv1(h)

        features = [h]
        for i, layer in enumerate(self.encoder):
            emb = logsnr_emb[..., None, None, :] + pose_embs[i]

            for block in layer:
                h = block(h, emb)
                features.append(h)

        emb = logsnr_emb[..., None, None, :] + pose_embs[-1]
        h = self.bottleneck(h)

        for i, layer in enumerate(self.decoder):
            emb = logsnr_emb[..., None, None, :] + pose_embs[-(i + 1)]
            for block in layer:
                feature = features.pop()
                h = torch.concat([h, feature], axis=1)
                h = block(h, emb)
