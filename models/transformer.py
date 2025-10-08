import torch
import torch.nn as nn
from models.mlp import MLP
from models.attention import MultiheadAttentionVarLen
from models.dense import Dense


class ContextProjector(nn.Module):
    def __init__(self, c_dim, model_dim, gated, activation="silu"):
        super().__init__()
        self.gated = gated
        self.out_factor = 2 * (1 + 1 + int(gated))  # scale, shift, (gate) x2
        self.c_proj = nn.Sequential(
            nn.SiLU() if activation == "silu" else nn.ReLU(),
            nn.Linear(c_dim, model_dim * self.out_factor)
        )
        ### Initial scale, shift, gate are 0
        nn.init.constant_(self.c_proj[1].weight, 0)
        nn.init.constant_(self.c_proj[1].bias, 0)

    def forward(self, c):
        affine_params = self.c_proj(c).unsqueeze(1).chunk(self.out_factor, dim=-1)
        return affine_params


class LayerScale(nn.Module):
    """Applies the LayerScale operation from the Cait vision transformer.

    Effective at improving stability and speed of deep transformers.
    Now the standard for vision transformers
    https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim: int, init_value: float = 1e-3) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTLayer(nn.Module):
    def __init__(self, embed_dim, context_dim, mha_config, dense_config=None, layer_scale=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.mha = MultiheadAttentionVarLen(embed_dim, **mha_config)

        if dense_config:
            self.dense = Dense(input_size=embed_dim, output_size=embed_dim, **dense_config)
        else:
            self.register_buffer("dense", None)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(context_dim, 6 * embed_dim, bias=True))

        if layer_scale:
            self.layer_scale_attn = LayerScale(embed_dim)
            self.layer_scale_mlp = LayerScale(embed_dim)

    def forward(self, q, q_mask=None, kv=None, kv_mask=None, context=None, attn_mask=None):
        '''
            if k is provided, then we will have cross-attention
        '''
        shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(context).chunk(6, dim=1)

        if kv == None: # self-attention
            q_attn = self.mha(
                q=modulate(self.norm1(q), shift_msa, scale_msa),
                q_mask=q_mask, attn_mask=attn_mask)

        else: # cross-attention
            q_attn = self.mha(
                q=q, kv=modulate(self.norm1(kv), shift_msa, scale_msa),
                q_mask=q_mask, kv_mask=kv_mask, attn_mask=attn_mask)
    
        if hasattr(self, 'layer_scale_attn'):
            q_attn = self.layer_scale_attn(q_attn)
        q = q + gate_msa.unsqueeze(1) * q_attn
        
        if self.dense:
            q_mlp = self.dense(modulate(self.norm2(q), shift_mlp, scale_mlp), context)
            if hasattr(self, 'layer_scale_mlp'):
                q_mlp = self.layer_scale_mlp(q_mlp)
            q = q + gate_mlp.unsqueeze(1) * q_mlp

        return q



class DiTEncoder(nn.Module):
    def __init__(
        self, embed_dim, num_layers, mha_config,
        dense_config=None, context_dim=0, out_dim=0, layer_scale=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        self.layers = nn.ModuleList(
            [DiTLayer(
                embed_dim, context_dim, 
                mha_config, dense_config, 
                layer_scale=layer_scale
            ) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        # For resizing the output tokens
        if self.out_dim:
            self.final_linear = nn.Linear(self.embed_dim, self.out_dim)


    def forward(self, q, **kwargs):
        for layer in self.layers:
            q = layer(q, **kwargs)
        q = self.final_norm(q)

        # Optinal resizing layer
        if self.out_dim:
            q = self.final_linear(q)
        return q



class DiTDecoder(nn.Module):
    def __init__(
        self, embed_dim, num_layers, sa_config, ca_config,
        dense_config=None, context_dim=0, out_dim=0, layer_scale=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        self.ca_layers = nn.ModuleList(
            [DiTLayer(
                embed_dim, context_dim, 
                ca_config, dense_config, 
                layer_scale=layer_scale,
            ) for _ in range(num_layers)]
        )
        self.sa_layers = nn.ModuleList(
            [DiTLayer(
                embed_dim, context_dim, 
                sa_config, dense_config, 
                layer_scale=layer_scale,
            ) for _ in range(num_layers)]
        )

        self.final_norm = nn.LayerNorm(embed_dim)

        # For resizing the output tokens
        if self.out_dim:
            self.final_linear = nn.Linear(self.embed_dim, self.out_dim)


    def forward(self, q, **kwargs):
        for sa_layer, ca_layer in zip(self.sa_layers, self.ca_layers):
            q = sa_layer(q, **kwargs)  # Self-attention layer
            q = ca_layer(q, **kwargs)  # Cross-attention layer

        q = self.final_norm(q)

        # Optinal resizing layer
        if self.out_dim:
            q = self.final_linear(q)
        return q
