import pytorch_lightning as pl
import torch
import torch.nn as nn
from onmt.encoders.transformer import TransformerEncoderLayer
from onmt.utils.misc import sequence_mask
from torch import FloatTensor, LongTensor
from typing import Tuple


class SparseConv(nn.Module):
    def __init__(self, kernel_size, h):
        super().__init__()

        self.kernel_size = kernel_size
        self.pooling_ratio = 4
        self.h = h
        self.in_channels = 1
        assert self.in_channels == 1, "Only supports greyscale images for now."

        self.embed = nn.Linear(kernel_size ** 2 * self.in_channels, h, bias=False)
        self.act = nn.GELU()

    def forward(self, x, x_lengths):
        B, L, _ = x.shape

        out = self.act(self.embed(x))           # (B, L, h)
        out = out.reshape(B, L // self.pooling_ratio, self.pooling_ratio,
                          -1)                   # (B, L/4, 4, h)
        out = torch.max(out, 2)[0]              # (B, L/4, h)

        out_lengths = x_lengths / self.pooling_ratio
        out_lengths = out_lengths.long()

        return out, out_lengths


class Encoder(pl.LightningModule):
    def __init__(self, d_model: int, kernel_size, enc_attn_layers: int, enc_attn_heads, enc_mlp_ratio):
        super().__init__()

        self.model = SparseConv(kernel_size=kernel_size, h=d_model)
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads=enc_attn_heads, d_ff=d_model*enc_mlp_ratio, dropout=0.2, attention_dropout=0.2,
                max_relative_positions=0)
             for i in range(enc_attn_layers)])

    def forward(self, imgs: FloatTensor, img_shapes: LongTensor, pixel_counts: LongTensor
                ) -> Tuple[FloatTensor, LongTensor]:
        """encode image to feature
        Returns
        -------
        Tuple[FloatTensor, LongTensor]
            [b, t, d], [b, t]
        """
        feature, lengths = self.model(imgs[..., 2:], pixel_counts)
        mask = ~sequence_mask(lengths)

        for layer in self.transformer:
            feature = layer(feature, mask.unsqueeze(1))
        # TODO: no post norm yet

        return feature, mask
