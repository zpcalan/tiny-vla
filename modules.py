import torch
from torch import nn

from diffusers.models.embeddings import SinusoidalPositionalEmbedding, Timesteps, TimestepEmbedding
from diffusers.models.attention import Attention, FeedForward

class TimeStepEmbed(nn.module):
    def __init__(self, cond_dim):
        self.emb = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.mlp = TimestepEmbedding(in_channels=256, time_embed_dim=cond_dim)

    def forward(self, timestep):
        return self.mlp(self.emb(timestep))
    

class AdaLN(nn.module):
    def __init__(self, cond_input_dim, hidden_dim):
        # Linear should output 2*hidden_dim for scale and shift tensor.
        self.silu = nn.SiLU()
        self.cond_proj = nn.Linear(cond_input_dim, hidden_dim*2)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, temb):
        output = self.cond_proj(self.silu(temb))
        scale, shift = output.chunk(2, dim=1)
        x = self.layer_norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


class TransformerBlock(nn.module):
    def __init__(self, hidden_dim, num_attn_heads, cond_dim):
        self.attn = Attention(query_dim=hidden_dim,
                              heads=num_attn_heads,
                              dim_head=hidden_dim/num_attn_heads,
                              cross_attention_dim=hidden_dim)
        self.ada_ln = AdaLN(cond_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.pos_embed = SinusoidalPositionalEmbedding(hidden_dim, 1024)
        self.mlp = FeedForward(hidden_dim)
    
    def forward(self, x, encoder_hidden_state, time_emb):
        residual = x
        # attn part
        x = self.ada_ln(x, time_emb)
        x = self.pos_embed(x)
        x = self.attn(x, encoder_hidden_state)
        x = x + residual

        # mlp part
        residual = x
        x = self.ln(x)
        x = self.mlp(x)
        x = x + residual