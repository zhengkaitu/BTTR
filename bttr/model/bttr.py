from typing import List

import pytorch_lightning as pl
import torch
from torch import FloatTensor, LongTensor

from bttr.utils import Hypothesis

from .decoder import Decoder
# from .encoder import Encoder
from .encoder_spt import Encoder


class BTTR(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        enc_attn_layers: int,
        enc_attn_heads: int,
        enc_mlp_ratio: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model,
            kernel_size=kernel_size,
            enc_attn_layers=enc_attn_layers,
            enc_attn_heads=enc_attn_heads,
            enc_mlp_ratio=enc_mlp_ratio
        )
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(
        self, imgs: FloatTensor, img_shapes: LongTensor, pixel_counts: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature, mask = self.encoder(imgs, img_shapes, pixel_counts)          # [b, t, d]
        feature = torch.cat((feature, feature), dim=0)                      # [2b, t, d]
        mask = torch.cat((mask, mask), dim=0)

        out = self.decoder(feature, mask, tgt)

        return out

    def beam_search(
        self, imgs: FloatTensor, img_shapes: LongTensor, pixel_counts: LongTensor, beam_size: int, max_len: int
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(imgs, img_shapes, pixel_counts)          # [1, t, d]
        return self.decoder.beam_search(feature, mask, beam_size, max_len)
