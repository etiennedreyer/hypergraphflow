import torch
import torch.nn as nn
import torch.nn.functional as F
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
        model_dim: int,
        c_dim: int = None,
        num_heads: int = 4,
        activation: str = "silu",
        gated: bool = False,
        scaling: bool = False,
        ffn_factor: int = 2,
        c_proj: nn.Module = None,
    ):
        super().__init__()

        self.model_dim = model_dim
        self.c_dim = c_dim
        self.num_heads = num_heads
        self.gated = gated
        self.scaling = scaling

        ### Check args
        assert not (scaling and gated), "scaling and gated should not both be True"
        assert not (c_dim is None and c_proj is not None), "c_proj requires c_dim != None"
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

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

        ### Head-mixing layer
        self.out_proj = nn.Linear(model_dim, model_dim)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0)

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
            self.alpha_attn = nn.Parameter(torch.tensor(1.0e-3))
            self.alpha_ffn = nn.Parameter(torch.tensor(1.0e-3))


    def modulate(self, x, scale, shift):
        return x * (1 + scale) + shift

    def get_qkv(self, x, y=None):
        """
        Returns:
            q, k, v: all of shape (batch_size, seq_len, model_dim)
        """
        pass

    def get_attn_mask(self, key_padding_mask=None, attn_mask=None):

        """
        Note: here, we use the convention that boolean attention mask is True for _masked_ pairs
        """

        if key_padding_mask is None and attn_mask is None:
            return None

        ### format key_padding_mask for multiple heads
        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool, "assumes boolean key_padding_mask"
            assert key_padding_mask.dim() == 2, "assumes batched key_padding_mask"
            key_padding_mask = key_padding_mask[:, None, None, :]

        ### format attn_mask for multiple heads
        if attn_mask is not None:
            assert attn_mask.dim() == 3, "assumes batched attn_mask"
            attn_mask = attn_mask[:, None, :, :]

        ### combine masks
        if key_padding_mask is not None and attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = attn_mask | key_padding_mask
            else:
                key_padding_mask = torch.where(key_padding_mask, float('-inf'), 0.0)
                attn_mask = attn_mask + key_padding_mask

        elif key_padding_mask is not None:
            attn_mask = key_padding_mask

        ### check for and hotfix fully masked queries
        if attn_mask.dtype == torch.bool:
            all_masked = attn_mask.all(dim=-1, keepdim=True)
            if all_masked.any():
                attn_mask = attn_mask & ~all_masked

        return attn_mask

    def attention(self, q, k, v, attn_mask=None):

        B, L, D = q.shape
        dot_dim = D // self.num_heads

        ### reshape qkv for multi-head attention
        reshape = lambda t: t.view(B, t.shape[1], self.num_heads, dot_dim).transpose(1,2)
        q = reshape(q)
        k = reshape(k)
        v = reshape(v)

        ### mha
        ### PyTorch's sdpa expects boolean attn_mask where "True indicates that the element should take part in attention."
        if (attn_mask is not None) and attn_mask.dtype == torch.bool:
            attn_mask = ~attn_mask
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)

        ### combine heads and project output
        attn = attn.transpose(1,2).reshape(B, L, D)
        attn = self.out_proj(attn)

        return attn

    def forward(self, x, y=None, c=None, key_padding_mask=None, attn_mask=None):
        """
        Forward pass for the AttentionLayer.

        Args:
            x: input tensor (batch_size, seq_len, model_dim)
            y: optional input for cross attention
            c: context tensor
            key_padding_mask: optional mask for padding
            attn_mask: optional attention mask for scaled_dot_product_attention
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

        ### combine key padding and attention masks
        attn_mask = self.get_attn_mask(key_padding_mask, attn_mask)

        ### multi-head attention
        attn = self.attention(q, k, v, attn_mask=attn_mask)

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
        super().__init__(*args, **kwargs)

        self.qkv_proj = nn.Linear(self.model_dim, 3*self.model_dim)
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.constant_(self.qkv_proj.bias, 0)

    def get_qkv(self, x, y=None):
        q, k ,v = self.qkv_proj(x).chunk(3, dim=-1)
        return q, k, v

 
class CrossAttentionLayer(AttentionLayer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.q_proj  = nn.Linear(self.model_dim,   self.model_dim)
        self.kv_proj = nn.Linear(self.model_dim, 2*self.model_dim)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.kv_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0)
        nn.init.constant_(self.kv_proj.bias, 0)

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
