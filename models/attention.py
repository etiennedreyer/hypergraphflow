import torch.nn as nn
from models.mlp import MLP

class Attention(nn.Module):
    def __init__(
        self,
        kind,
        input_dim_kv,
        model_dim,
        c_dim=None,
        input_dim_q=None,
        num_heads=4,
        num_layers=3,
        batch_first=True,
        activation="silu",
    ):
        super().__init__()

        assert kind in ["self", "cross"], f"kind must be 'self' or 'cross', got {kind}"

        self.input_dim_kv = input_dim_kv
        self.model_dim = model_dim
        self.input_dim_q = input_dim_q if input_dim_q is not None else input_dim_kv
        self.c_dim = c_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.mha_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=model_dim, 
                                  num_heads=num_heads, 
                                  batch_first=batch_first)
            for _ in range(self.num_layers)
        ])

        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(model_dim,
                         elementwise_affine=(c_dim is None),
                         ) for _ in range(num_layers)
        ])

        ### Context embedding projection to modulate LayerNorms (scale, shift)
        if c_dim is not None:
            self.c_proj1 = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU() if activation == "silu" else nn.ReLU(),
                    nn.Linear(c_dim, model_dim * 2)
                )
                for _ in range(num_layers)
            ])

        self.ffn_layers = nn.ModuleList([
            MLP(model_dim, 
                [model_dim * 2], 
                model_dim,
                activation=activation)
            for _ in range(num_layers)
        ])

        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(model_dim,
                        elementwise_affine=(c_dim is None),
                        ) for _ in range(num_layers)
        ])

        if c_dim is not None:
            self.c_proj2 = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU() if activation == "silu" else nn.ReLU(),
                    nn.Linear(c_dim, model_dim * 2)
                )
                for _ in range(num_layers)
            ])

    def modulate(self, x, scale, shift):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def get_qkv(self, x, y=None):
        """
        Returns:
            q, k, v: all of shape (batch_size, seq_len, model_dim)
        """
        pass

    def forward(self, x, y=None, c=None, key_padding_mask=None):
        
        assert self.c_dim is None or c is not None, \
            "context c must be provided if c_dim is not None!"

        ### assumes x is of shape (batch_size, seq_len, input_dim)
        q, k, v = self.get_qkv(x, y)

        if not self.batch_first:
            q = q.permute(1, 0, 2)
            k = k.permute(1, 0, 2)
            v = v.permute(1, 0, 2)

        for i in range(self.num_layers):
            attn = self.mha_layers[i](q, k, v, 
                                      key_padding_mask=key_padding_mask
                                      )[0]

            ### residual + norm 1
            q = self.norm1_layers[i](q + attn)

            if self.c_dim is not None:
                ### context modulation 1
                scale1, shift1 = self.c_proj1[i](c).chunk(2, dim=-1)
                q = self.modulate(q, scale1, shift1)

            ### feed forward
            ffn_out = self.ffn_layers[i](q)

            ### residual + norm 2
            q = self.norm2_layers[i](q + ffn_out)

            if self.c_dim is not None:
                ### context modulation 2
                scale2, shift2 = self.c_proj2[i](c).chunk(2, dim=-1)
                q = self.modulate(q, scale2, shift2)

        return q
    

class SelfAttention(Attention):
    
    def __init__(self, *args, **kwargs):
        super().__init__(kind="self", *args, **kwargs)

        self.qkv_proj = nn.Linear(self.input_dim_q, 3*self.model_dim)

    def get_qkv(self, x, y=None):
        q, k ,v = self.qkv_proj(x).chunk(3, dim=-1)
        return q, k, v

 
class CrossAttention(Attention):

    def __init__(self, *args, **kwargs):
        super().__init__(kind="cross", *args, **kwargs)

        self.q_proj  = nn.Linear(self.input_dim_q,    self.model_dim)
        self.kv_proj = nn.Linear(self.input_dim_kv, 2*self.model_dim)

    def get_qkv(self, x, y):
        q = self.q_proj(x)
        k, v = self.kv_proj(y).chunk(2, dim=-1)
        return q, k, v