import torch
import torch.nn as nn
from math import log


def gen_positional_encoding(d_model : int, max_seq_len : int) -> torch.Tensor:
    """
    Precompute a positional encoding tensor (trig encoding from 'Attention is All You Need').

    Args:
        d_model: The input representation dimension (like embedding dimension)
        max_seq_len: The maximum length of PEs to precompute (normally large to accomodate long sequences).
    """
    position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
    pe = torch.zeros((max_seq_len, d_model))
    # apply sin terms to even indices
    pe[:, 0::2] = torch.sin(position * div_term)
    # apply cos terms to odd indices
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class Transformer(nn.Module):
    def __init__(
            self,
            d_model : int,
            max_seq_len : int = 256,
            d_in : int = 1,
            d_out : int = 1,
            n_head : int = 8,
            dim_feedforward : int = 2048,
            dropout : float = 0,
            n_encoder_layers = 1
        ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.register_buffer('pe', gen_positional_encoding(d_model, max_seq_len))

        self.input_projection = nn.Linear(d_in, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=n_encoder_layers
        )
        self.decoder = nn.Linear(d_model, d_out)

    def toggle_dropout(self, cond):
        attr = 'train' if cond else 'eval'
        getattr(self.encoder_layer.dropout, attr)()
        getattr(self.encoder_layer.dropout1, attr)()
        getattr(self.encoder_layer.dropout2, attr)()
        getattr(self.encoder_layer, attr)()
        getattr(self.encoder_layer.self_attn, attr)()
        getattr(self.encoder, attr)()

    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x : torch.Tensor):
        seq_len = x.size(1)
        causal_mask = self._generate_causal_mask(seq_len).to(x.device)
        x = self.input_projection(x)
        assert x.size(1) <= self.max_seq_len
        x += self.pe[:x.size(1), :]
        x = self.encoder(x, mask=causal_mask)
        x = self.decoder(x)
        return x
