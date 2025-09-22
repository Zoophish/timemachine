import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEncoding(nn.Module):
    """
    Method from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    def __init__(self, dim, device='cpu', precompute_seq_len=512, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.device = device

        # precompute inverse of theta vector Θ = {θi = 10000−2(i−1)/d, i ∈ [1, 2, ..., d/2]}
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim)).to(device)
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=precompute_seq_len, device=device)

    def _set_cos_sin_cache(self, seq_len, device):
        t = torch.arange(seq_len, device=device)
        
        # calculate the arguments for sin and cos: m * theta_i
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # concatenate along the last dimension to get pairs of (m*theta_i, m*theta_i)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_cached = emb.cos()
        sin_cached = emb.sin()
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self.max_seq_len_cached = seq_len

    def forward(self, x : torch.Tensor):
        batch_size, n_head, seq_len, dim = x.shape
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
        # add singleton dimensions for batch and attn heads
        cos = self.cos_cached[None, None, :seq_len, ...]
        sin = self.sin_cached[None, None, :seq_len, ...]

        # computationally efficient form of rotary matrix multiplication
        x_half_1 = x[..., 0::2]  # get every even indexed element
        x_half_2 = x[..., 1::2]  # ... and every odd indexed
        # x_rotated = torch.cat((-x_half_2, x_half_1), dim=-1)
        x_rotated = torch.stack([-x_half_2, x_half_1], dim=-1).flatten(start_dim=-2)
        return x * cos + x_rotated * sin


class Attention(nn.Module):
    def __init__(self, dim : int, n_head : int, device='cpu', use_rope : bool = True):
        super().__init__()

        if dim % n_head != 0:
            raise ValueError("dim must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.device = device
        
        # query, key, value, output matrices
        self.W_q = nn.Linear(dim, dim, bias=False, device=device)
        self.W_k = nn.Linear(dim, dim, bias=False, device=device)
        self.W_v = nn.Linear(dim, dim, bias=False, device=device)
        self.W_o = nn.Linear(dim, dim, bias=False, device=device)
        self.rope = RotaryPositionalEncoding(self.head_dim, device) if use_rope else None

    def forward(self, x : torch.Tensor, mask=None):
        batch_size, seq_len, n_feat = x.shape

        # generate query, key and value for every token in seq
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        # separate each head and transpose for correct batching
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        # apply RoPE
        if self.rope:
            q = self.rope(q)
            k = self.rope(k)
        # dot product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.W_o(output)
        return output


class EncoderLayer(nn.Module):
    def __init__(
            self,
            d_model : int,
            n_head : int,
            d_ff : int,
            dropout : float,
            device = 'cpu',
            use_rope = True
        ):
        super().__init__()
        self.attention = Attention(
            d_model,
            n_head,
            device=device, 
            use_rope=use_rope
        )
        self.fc1 = nn.Linear(d_model, d_ff, device=device)
        self.fc2 = nn.Linear(d_ff, d_model, device=device)
        self.activation = nn.GELU()
        self.layer_norm1 = nn.LayerNorm(d_model, device=device)
        self.layer_norm2 = nn.LayerNorm(d_model, device=device)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x : torch.Tensor, mask=None):
        # apply pre-layer normalisation
        normed_x = self.layer_norm1(x)
        attn = self.attention(normed_x, mask)
        x = x + self.dropout(attn)

        normed_x = self.layer_norm2(x)
        ff_out = self.fc2(self.activation(self.fc1(normed_x)))
        x = x + self.dropout(ff_out)
        return x


class DecoderTransformer(nn.Module):
    def __init__(
            self,
            in_dim : int,
            out_dim : int,
            d_model : int,
            n_head : int,
            d_ff : int,
            dropout : float,
            n_layers : int,
            device = 'cpu',
            use_rope = True
        ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.input_projection = nn.Linear(in_dim, d_model, device=device)
        self.decoder_blocks = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout, device, use_rope)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, out_dim, device=device)

    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        # the causal mask will mean that at any output position, the attention heads
        # can only attend to inputs preceding it. This is a training trick which is
        # more efficient than showing the model input/target pairs
        mask = torch.triu(torch.ones(sz, sz, device=self.device), diagonal=1).bool()
        return mask
    
    def forward(self, x : torch.Tensor):
        seq_len = x.size(1)
        causal_mask = self._generate_causal_mask(seq_len)

        x = self.input_projection(x)

        decoder_out = x
        for decoder_block in self.decoder_blocks:
            decoder_out = decoder_block(decoder_out, causal_mask)
        
        out = self.fc(decoder_out)
        return out
