import argparse
import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
from onmt.encoders.transformer import TransformerEncoderLayer


def get_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--enc_depths", type=str, default="2,2,2,2,8")
    parser.add_argument("--enc_num_heads", type=str, default="8,8,8,8,8")
    parser.add_argument("--enc_emb_size", type=int, default=256)
    parser.add_argument("--enc_hidden_sizes", type=str, default="256,256,256,256,256")
    parser.add_argument("--enc_mlp_ratio", type=int, default=4)
    parser.add_argument("--enc_window_size", help="Encoder window size", type=int, default=32)
    parser.add_argument('--swin_attention', action="store_true", default=True)
    parser.add_argument('--max_pixels', type=int, default=4096)
    parser.add_argument('--cross_level', type=int, default=2)
    parser.add_argument('--pool', type=str, default="cat")
    parser.add_argument('--cross_attn_pe', type=str, default="rel_emb")
    parser.add_argument("--hidden_dropout", help="Hidden dropout", type=float, default=0.1)
    parser.add_argument("--attn_dropout", help="Attention dropout", type=float, default=0.1)
    parser.add_argument("--max_relative_positions", help="Max relative positions", type=int, default=0)
    args, _ = parser.parse_known_args()

    return args


class PositionProjection(nn.Module):
    def __init__(self, args, h: int):
        super().__init__()
        self.args = args
        self.proj_1 = nn.Linear(2, h, bias=True)
        self.proj_2 = nn.Linear(h, h, bias=True)
        self.layer_norm_1 = nn.LayerNorm(h, eps=1e-6)
        self.dropout = nn.Dropout(args.hidden_dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        h = self.dropout(self.gelu(self.layer_norm_1(self.proj_1(x))))
        h = self.dropout(self.gelu((self.proj_2(h))))                       # (B, L, h)

        return h


class MaxPool(nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super().__init__()

        self.ffn = nn.Linear(in_channels, out_channels, bias=True)
        self.layer_norm = nn.LayerNorm(out_channels, eps=1e-6)
        self.dropout = nn.Dropout(args.hidden_dropout)
        self.gelu = nn.GELU()

        self.scale_factor = out_channels // in_channels

    def forward(self, x, p=None):
        """
            x: (B, L, h)
            out: (B, L/2, h)
        """
        batch_size, patch_count, hidden_size = x.shape
        h = self.dropout(self.gelu(self.layer_norm(self.ffn(x))))           # (B, L, h)
        h = h.reshape(batch_size, patch_count // 2, 2, -1)                  # (B, L/2, 2, h)
        h = torch.max(h, 2)[0]                                              # (B, L/2, h)
        xmax = torch.max(
            x.reshape(batch_size, patch_count // 2, 2, hidden_size), 2)[0]
        xmax = xmax.repeat(1, 1, self.scale_factor)
        h = h + xmax                                                        # residual

        return h, p


class CatPool(nn.Module):
    def __init__(self, args, h_in, h_out):
        super().__init__()

        self.ffn = nn.Linear(h_in * 2, h_out, bias=True)
        self.layer_norm = nn.LayerNorm(h_out, eps=1e-6)
        self.dropout = nn.Dropout(args.hidden_dropout)
        self.gelu = nn.GELU()

        self.scale_factor = h_out // h_in
        self.pe = PositionProjection(args, h=h_in * 2)

    def forward(self, x: torch.Tensor, p: torch.Tensor):
        """
            x: (B, L, h)
            p: (B, L, 2)
            out: (B, L/2, ...)
        """
        batch_size, patch_count, hidden_size = x.shape
        x = x.reshape(batch_size, patch_count // 2, hidden_size * 2)

        p = p.reshape(batch_size, patch_count // 2, 2, 2)
        p_out = torch.mean(p, dim=2)                                        # -> (B, L/2, 2), getting the centroid
        p_diff = p[:, :, 1, :] - p[:, :, 0, :]
        rel_pe = self.pe(p_diff)

        h = self.dropout(self.gelu(self.layer_norm(self.ffn(x + rel_pe))))  # (B, L/2, h)
        xmax = torch.max(
            x.reshape(batch_size, patch_count // 2, 2, hidden_size), 2)[0]
        xmax = xmax.repeat(1, 1, self.scale_factor)
        h = h + xmax                                                        # residual

        return h, p_out


class CrossAttention(nn.Module):
    def __init__(self, args, heads: int, h: int, shifted: int = 0):
        super().__init__()
        self.args = args
        self.shifted = shifted
        self.pe = None
        self.pe_type = args.cross_attn_pe

        self.model_dim = h
        self.head_count = heads
        self.dim_per_head = self.model_dim // self.head_count
        assert self.dim_per_head == 32, \
            f"dim_per_head should be 32, got {self.dim_per_head} instead"
        self.window_size = args.enc_window_size

        if self.pe_type == "rel_emb":
            self.pe = PositionProjection(args, h=h)
            self.c = nn.Parameter(torch.randn(h), requires_grad=True)
            self.d = nn.Parameter(torch.randn(h), requires_grad=True)
        elif self.pe_type == "abs_emb":
            self.pe = PositionProjection(args, h=h)

        self.linear_q = nn.Linear(self.model_dim, self.model_dim, bias=True)
        self.linear_k = nn.Linear(self.model_dim, self.model_dim, bias=True)
        self.linear_v = nn.Linear(self.model_dim, self.model_dim, bias=True)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.attn_dropout)
        self.final_linear = nn.Linear(self.model_dim, self.model_dim)

    def forward(self, inputs, p, mask=None):
        assert mask is None

        # print(f"inputs shape: {inputs.shape}")
        B, L, C = inputs.shape
        n_window = L // self.window_size
        X = self.args.cross_level

        def _shape(x, shifted: int):
            """Projection, possibly shifted"""
            if shifted == 0:
                _shaped = x
            elif shifted == 1:
                indices = torch.arange(n_window * 2 ** X, device=x.device)
                indices = indices.reshape(n_window * 2 ** (X - 1), 2)
                indices = indices.transpose(0, 1).reshape(n_window * 2 ** X)
                _shaped = x.reshape(B, n_window * 2 ** X, self.window_size // 2 ** X, C)
                _shaped = _shaped[:, indices, :, :]                             # shifted indexing
            else:
                raise NotImplementedError

            _shaped = _shaped.reshape(B, n_window, self.window_size, self.head_count, self.dim_per_head)
            _shaped = _shaped.transpose(2, 3)
            return _shaped

        def _unshape(x, shifted: int):
            """Concatenate by window and by head, possibly shifted"""
            _unshaped = x.transpose(2, 3).contiguous()                          # (B, L // M, M, head, C/head)

            if shifted == 0:
                pass
            elif shifted == 1:
                indices = torch.arange(n_window * 2 ** X, device=x.device)
                indices = indices.reshape(2, n_window * 2 ** (X - 1))
                indices = indices.transpose(0, 1).reshape(n_window * 2 ** X)
                _unshaped = _unshaped.reshape(B, n_window * 2 ** X, self.window_size // 2 ** X, C)
                _unshaped = _unshaped[:, indices, :, :]                         # undo shifted indexing
            else:
                raise NotImplementedError

            _unshaped = _unshaped.reshape(B, L, C)
            return _unshaped

        def _shape_p(_p, shifted: int):
            """Projection, possibly shifted"""
            if shifted == 0:
                _shaped = _p
            elif shifted == 1:
                indices = torch.arange(n_window * 2 ** X, device=_p.device)
                indices = indices.reshape(n_window * 2 ** (X - 1), 2)
                indices = indices.transpose(0, 1).reshape(n_window * 2 ** X)
                _shaped = _p.reshape(B, n_window * 2 ** X, self.window_size // 2 ** X, 2)
                _shaped = _shaped[:, indices, :, :]                             # shifted indexing
            else:
                raise NotImplementedError

            _shaped = _shaped.reshape(B, n_window, self.window_size, 2)
            return _shaped

        if self.pe_type == "abs_emb":
            # inputs = inputs * math.sqrt(self.model_dim)
            inputs = inputs + self.pe(p)
            # inputs = self.dropout(inputs)

        q = self.linear_q(inputs)
        k = self.linear_k(inputs)
        v = self.linear_v(inputs)                               # (B, L, C)

        # each window SA itself, similarly for each head
        q = _shape(q, self.shifted) / math.sqrt(self.dim_per_head)
        k = _shape(k, self.shifted)                             # (B, L // M, head, M_k, C/head)
        v = _shape(v, self.shifted)                             # (B, L // M, head, M_k, C/head)

        if self.pe_type == "rel_emb":
            # a + c
            c = self.c.reshape(1, 1, self.head_count, 1, self.dim_per_head)
            a_c = torch.matmul(q + c, k.transpose(3, 4))

            # relative pe
            p = _shape_p(p, self.shifted)                           # (B, L // M, M_q, 2)
            pe = self.pe(p)                                         # (B, L // M, M_q, h)
            pe_exp = pe.unsqueeze(-2)                               # (B, L // M, M_q, 1, h)
            pe_exp = pe_exp.repeat(1, 1, 1, self.window_size, 1)    # (B, L // M, M_q, M_k, h)
            pe = pe_exp - pe.unsqueeze(-3)                          # (B, L // M, M_q, M_k, h)

            pe = pe.reshape(
                B, n_window, self.window_size, self.window_size, self.head_count, self.dim_per_head)

            # b + d
            q = q.unsqueeze(-2)                                 # -> (B, L // M, head, M_q, 1, C/head)
            pe_t = pe.permute(                                  # (B, L // M, M_q, M_k, head, C/head)
                0, 1, 4, 2, 5, 3)                               # -> (B, L // M, head, M_q, C/head, M_k)

            d = self.d.reshape(1, 1, self.head_count, 1, 1, self.dim_per_head)
            b_d = torch.matmul(q + d, pe_t                      # (B, L // M, head, M_q, 1, M_k)
                               ).squeeze(-2)                    # -> (B, L // M, head, M_q, M_k)

            scores = a_c + b_d

        else:
            qk = torch.matmul(q, k.transpose(3, 4))             # (B, L // M, head, M_q, M_k)
            scores = qk

        scores = scores.float()

        attn = self.softmax(scores).to(q.dtype)
        drop_attn = self.dropout(attn)

        context = torch.matmul(drop_attn, v)                    # (B, L // M, head, M, C/head)
        context = _unshape(context, self.shifted)               # (B, L, C)
        output = self.final_linear(context)

        return output


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.

    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """Layer definition.
            Args: x: ``(batch_size, input_len, model_dim)``
            Returns: (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.dropout_1(self.gelu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class PixelSwinLayer(nn.Module):
    def __init__(self, args, heads: int, h: int, pool="none", attn="cross"):
        super().__init__()
        self.args = args
        self.d_model = h
        self.pooling = None

        if attn == "cross":
            self.self_attn_w = CrossAttention(args, heads, h, shifted=0)
            self.self_attn_sw = CrossAttention(args, heads, h, shifted=args.swin_attention * 1)
        else:
            raise NotImplementedError

        self.feed_forward_w = PositionwiseFeedForward(
            d_model=h,
            d_ff=h * args.enc_mlp_ratio,
            dropout=args.hidden_dropout
        )
        self.feed_forward_sw = PositionwiseFeedForward(
            d_model=h,
            d_ff=h * args.enc_mlp_ratio,
            dropout=args.hidden_dropout
        )
        self.layer_norm_w = nn.LayerNorm(h, eps=1e-6)
        self.layer_norm_sw = nn.LayerNorm(h, eps=1e-6)
        self.dropout = nn.Dropout(args.hidden_dropout)

        if pool == "max":
            self.pooling = MaxPool(args, self.d_model, self.d_model)
        elif pool == "max4":
            self.pooling = MaxPool(args, self.d_model, self.d_model * 2)
        elif pool == "cat":
            self.pooling = CatPool(args, self.d_model, self.d_model)
        else:
            assert pool == "none"

    def forward(self, inputs, p, mask=None):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            p (FloatTensor): ``(batch_size, src_len, 2)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len // 2, model_dim)`` if pool == "max"
                    or ``(batch_size, src_len // 4, model_dim * 2)`` if pool == "max"
                    or ``(batch_size, src_len, model_dim)`` if pool == None
        """
        input_norm = self.layer_norm_w(inputs)
        context = self.self_attn_w(input_norm, p, mask=mask)
        out = self.dropout(context) + inputs
        out = self.feed_forward_w(out)

        input_norm = self.layer_norm_sw(out)
        context = self.self_attn_sw(input_norm, p, mask=mask)
        out = self.dropout(context) + out
        out = self.feed_forward_sw(out)

        if self.pooling is not None:
            out, p = self.pooling(out, p)

        return out, p


class SPTCross(nn.Module):
    """
        Returns: torch.FloatTensor: * memory_bank ``(batch_size, src_len, model_dim)``
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.transformer_swin = nn.ModuleList()
        self.transformer_global = nn.ModuleList()
        self.n_group = len(args.enc_depths)

        for i in range(self.n_group - 1):
            d = args.enc_depths[i]
            n = args.enc_num_heads[i]
            h = args.enc_hidden_sizes[i]
            assert d % 2 == 0
            n_block = d // 2
            for j in range(n_block - 1):
                layer = PixelSwinLayer(args, heads=n, h=h, pool="none", attn="cross")
                self.transformer_swin.append(layer)

            # pool only at the end
            layer = PixelSwinLayer(args, heads=n, h=h, pool=args.pool, attn="cross")
            self.transformer_swin.append(layer)

        d = args.enc_depths[-1]
        n = args.enc_num_heads[-1]
        h = args.enc_hidden_sizes[-1]
        assert d % 2 == 0
        n_block = d // 2
        for j in range(n_block):
            # layer = PixelSwinLayer(args, heads=n, h=h, pool="none", attn="local")
            layer = TransformerEncoderLayer(
                d_model=h,
                heads=n,
                d_ff=h*args.enc_mlp_ratio,
                dropout=args.hidden_dropout,
                attention_dropout=args.attn_dropout,
                max_relative_positions=0
            )

            self.transformer_global.append(layer)

        self.layer_norm = nn.LayerNorm(h, eps=1e-6)

    def forward(self, src, p, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        out = src

        for layer in self.transformer_swin:
            out, p = layer(out, p, mask=None)
        for layer in self.transformer_global:
            out = layer(out, mask=None)

        out = self.layer_norm(out)

        return out


class Encoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        args = get_args()
        args.enc_depths = [int(d) for d in args.enc_depths.split(",")]
        args.enc_num_heads = [int(n) for n in args.enc_num_heads.split(",")]
        args.enc_hidden_sizes = [int(h) for h in args.enc_hidden_sizes.split(",")]
        assert len(args.enc_depths) == len(args.enc_num_heads) == len(args.enc_hidden_sizes)

        self.pos_projection = PositionProjection(args, h=args.enc_emb_size)
        self.encoder = SPTCross(args)

    def forward(self, images):
        """images: (b, t, 2)"""
        embedded_positions = self.pos_projection(images)                # (b, t, 2) -> (b, t, h)
        memory_bank = self.encoder(embedded_positions, images)

        return memory_bank
