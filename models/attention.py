import torch
import torch.nn as nn
from models.mlp import MLP


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


class AttentionLayer(nn.Module):
    def __init__(
        self,
        kind: str, # "self" or "cross"
        model_dim: int,
        c_dim: int = None,
        num_heads: int = 4,
        batch_first: bool = True,
        activation: str = "silu",
        gated: bool = False,
        scaling: bool = False,
        ffn_factor: int = 2,
        c_proj: nn.Module = None,
    ):
        super().__init__()

        assert kind in ["self", "cross"], f"kind must be 'self' or 'cross', got {kind}"

        self.model_dim = model_dim
        self.c_dim = c_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.gated = gated
        self.scaling = scaling

        ### Check args
        assert not (scaling and gated), "scaling and gated should not both be True"
        assert not (c_dim is None and c_proj is not None), "c_proj requires c_dim != None"

        ### First norm
        self.norm1 = nn.LayerNorm(
                            model_dim,
                            elementwise_affine=(c_dim is None),
                        )

        ### Context embedding projection to modulate and gate query
        if c_dim is not None:
            if c_proj is None:
                self.c_proj = ContextProjector(c_dim, model_dim, \
                                               gated, activation=activation)
            else:
                self.c_proj = c_proj # shared weights for context projection

        ### Multi-head attention layers
        self.mha = nn.MultiheadAttention(
                                  embed_dim=model_dim, 
                                  num_heads=num_heads, 
                                  batch_first=batch_first
                                )

        ### Feed forward network
        self.ffn = MLP(
                    model_dim, 
                    [model_dim * ffn_factor],
                    model_dim,
                    activation=activation
                )

        ### Second norm
        self.norm2 = nn.LayerNorm(
                            model_dim,
                            elementwise_affine=(c_dim is None),
                        )

        # Scaling parameters, initialized to zero
        if scaling:
            self.alpha_attn = nn.Parameter(torch.tensor(0.0))
            self.alpha_ffn = nn.Parameter(torch.tensor(0.0))

    def modulate(self, x, scale, shift):
        return x * (1 + scale) + shift

    def get_qkv(self, x, y=None):
        """
        Returns:
            q, k, v: all of shape (batch_size, seq_len, model_dim)
        """
        pass

    def forward(self, x, y=None, c=None, key_padding_mask=None, attn_mask=None):
        """
        Forward pass for the AttentionLayer.

        Args:
            x: input tensor (batch_size, seq_len, model_dim)
            y: optional input for cross attention
            c: context tensor
            key_padding_mask: optional mask for padding
            attn_mask: optional attention mask for MultiheadAttention
        """
        assert self.c_dim is None or c is not None, \
            "context c must be provided if c_dim is not None!"

        ### prenorm x and y for qkv projection
        x_norm = self.norm1(x)
        y_norm = self.norm1(y) if y is not None else None

        if self.c_dim is not None:
            affine_params = self.c_proj(c)
            ### unpack parameters from context projection
            if self.gated:
                scale1, shift1, gate1, scale2, shift2, gate2 = affine_params
            else:
                scale1, shift1, scale2, shift2 = affine_params

            ### context modulation 1
            x_norm = self.modulate(x_norm, scale1, shift1)

        ### assumes x is of shape (batch_size, seq_len, model_dim)
        q, k, v = self.get_qkv(x_norm, y_norm)

        if not self.batch_first:
            q = q.permute(1, 0, 2)
            k = k.permute(1, 0, 2)
            v = v.permute(1, 0, 2)

        ### multi-head attention
        attn = self.mha(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]

        if self.gated and self.c_dim is not None:
            ### gate attention
            attn = gate1 * attn
        elif self.scaling:
            attn = self.alpha_attn * attn

        ### residual + attention
        x = x + attn

        ### prenorm x for ffn
        x_norm = self.norm2(x)

        if self.c_dim is not None:
            ### context modulation 2
            x_norm = self.modulate(x_norm, scale2, shift2)

        ### feed forward
        ffn_out = self.ffn(x_norm)

        if self.gated and self.c_dim is not None:
            ### gate ffn
            ffn_out = gate2 * ffn_out
        elif self.scaling:
            ffn_out = self.alpha_ffn * ffn_out

        ### residual + ffn
        x = x + ffn_out

        return x
    

class SelfAttentionLayer(AttentionLayer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(kind="self", *args, **kwargs)

        self.qkv_proj = nn.Linear(self.model_dim, 3*self.model_dim)

    def get_qkv(self, x, y=None):
        q, k ,v = self.qkv_proj(x).chunk(3, dim=-1)
        return q, k, v

 
class CrossAttentionLayer(AttentionLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(kind="cross", *args, **kwargs)

        self.q_proj  = nn.Linear(self.model_dim,   self.model_dim)
        self.kv_proj = nn.Linear(self.model_dim, 2*self.model_dim)

    def get_qkv(self, x, y):
        q = self.q_proj(x)
        k, v = self.kv_proj(y).chunk(2, dim=-1)
        return q, k, v

class DecoderBlock(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.SA = SelfAttentionLayer(*args, **kwargs)
        self.CA = CrossAttentionLayer(*args, **kwargs)
        self.num_heads = self.CA.num_heads

    def forward(self, x_a, x_b, c=None, key_padding_mask_SA=None, key_padding_mask_CA=None,
                attn_mask_SA=None, attn_mask_CA=None):
        """
        Forward pass for DecoderBlock.

        Args:
            x_a: input tensor (batch_size, seq_len, model_dim)
            x_b: cross input tensor
            c: context tensor
            key_padding_mask_SA: key padding mask for self-attention
            key_padding_mask_CA: key padding mask for cross-attention
            attn_mask_SA: attention mask for self-attention
            attn_mask_CA: attention mask for cross-attention
        """
        ### Self attention
        x_a = self.SA(x_a, c=c, key_padding_mask=key_padding_mask_SA, attn_mask=attn_mask_SA)

        ### Cross attention
        x_a = self.CA(x_a, x_b, c=c, key_padding_mask=key_padding_mask_CA, attn_mask=attn_mask_CA)

        return x_a

class DualUpdateBlock(nn.Module):

    def __init__(self, *args, **kwargs):
        
        super().__init__()
        self.CA_a = CrossAttentionLayer(*args, **kwargs)
        self.CA_b = CrossAttentionLayer(*args, **kwargs)
        self.num_heads = self.CA_a.num_heads

    def forward(self, x_a, x_b, c=None, key_padding_mask_a=None, key_padding_mask_b=None, attn_mask_a=None, attn_mask_b=None):

        ### First update
        # q: x_a
        # k/v: x_b
        x_a = self.CA_a(x_a, x_b, c=c, key_padding_mask=key_padding_mask_b, 
                        attn_mask=attn_mask_a)

        ### Second update
        # q: x_b
        # k/v: x_a
        x_b = self.CA_b(x_b, x_a, c=c, key_padding_mask=key_padding_mask_a, 
                        attn_mask=attn_mask_b)

        return x_a, x_b
