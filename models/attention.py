import torch
import torch.nn as nn
from models.mlp import MLP

class AttentionLayer(nn.Module):
    def __init__(
        self,
        kind,
        model_dim,
        c_dim=None,
        num_heads=4,
        batch_first=True,
        activation="silu",
        gated=False,
    ):
        super().__init__()

        assert kind in ["self", "cross"], f"kind must be 'self' or 'cross', got {kind}"

        self.model_dim = model_dim
        self.c_dim = c_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.gated = gated

        ### First norm
        self.norm1 = nn.LayerNorm(
                            model_dim,
                            elementwise_affine=(c_dim is None),
                        )

        ### Context embedding projection to modulate and gate query
        if c_dim is not None:
            self.c_proj = nn.Sequential(
                nn.SiLU() if activation == "silu" else nn.ReLU(),
                nn.Linear(c_dim, model_dim * (4 + 2*int(gated))) # scale, shift, (gate)
            )

            ### Initial scale, shift, gate are 0
            nn.init.constant_(self.c_proj[1].weight, 0)
            nn.init.constant_(self.c_proj[1].bias, 0)

        ### Multi-head attention layers
        self.mha = nn.MultiheadAttention(
                                  embed_dim=model_dim, 
                                  num_heads=num_heads, 
                                  batch_first=batch_first
                                )

        ### Feed forward network
        self.ffn = MLP(
                    model_dim, 
                    [model_dim * 2], 
                    model_dim,
                    activation=activation
                )

        ### Second norm
        self.norm2 = nn.LayerNorm(
                            model_dim,
                            elementwise_affine=(c_dim is None),
                        )

    def modulate(self, x, scale, shift):
        return x * (1 + scale) + shift

    def get_qkv(self, x, y=None):
        """
        Returns:
            q, k, v: all of shape (batch_size, seq_len, model_dim)
        """
        pass

    def forward(self, x, y=None, c=None, key_padding_mask=None):
        
        assert self.c_dim is None or c is not None, \
            "context c must be provided if c_dim is not None!"

        ### assumes x is of shape (batch_size, seq_len, model_dim)
        q, k, v = self.get_qkv(x, y)

        if not self.batch_first:
            q = q.permute(1, 0, 2)
            k = k.permute(1, 0, 2)
            v = v.permute(1, 0, 2)

        ### prenorm query for mha
        q_norm = self.norm1(q)

        if self.c_dim is not None:

            ### unpack parameters from context projection
            if self.gated:
                scale1, shift1, gate1, \
                    scale2, shift2, gate2 = self.c_proj(c).unsqueeze(1).chunk(6, dim=-1)
            else:
                scale1, shift1, \
                    scale2, shift2 = self.c_proj(c).unsqueeze(1).chunk(4, dim=-1)

            ### context modulation 1
            q_norm = self.modulate(q_norm, scale1, shift1)

        ### multi-head attention
        attn = self.mha(q_norm, k, v, key_padding_mask=key_padding_mask)[0]

        if self.gated and self.c_dim is not None:
            ### gate attention
            attn = gate1 * attn

        ### residual + attention
        q = q + attn

        ### prenorm query for ffn
        q_norm = self.norm2(q)

        if self.c_dim is not None:
            ### context modulation 2
            q_norm = self.modulate(q_norm, scale2, shift2)

        ### feed forward
        ffn_out = self.ffn(q_norm)

        if self.gated and self.c_dim is not None:
            ### gate ffn
            ffn_out = gate2 * ffn_out

        ### residual + ffn
        q = q + ffn_out

        return q
    

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
    
class DualUpdateBlock(nn.Module):

    def __init__(self, *args, **kwargs):
        
        super().__init__()
        self.CA_a = CrossAttentionLayer(*args, **kwargs)
        self.CA_b = CrossAttentionLayer(*args, **kwargs)

    def forward(self, x_a, x_b, c=None, key_padding_mask=None):

        ### First update
        # q: x_a
        # k/v: x_b
        x_a = self.CA_a(x_a, x_b, c=c, key_padding_mask=key_padding_mask)

        ### Second update
        # q: x_b
        # k/v: x_a
        x_b = self.CA_b(x_b, x_a, c=c, key_padding_mask=key_padding_mask)

        return x_a, x_b
