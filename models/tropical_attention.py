import torch
import torch.nn as nn
import torch.nn.functional as F


### Source: https://github.com/Baran-phys/Tropical-Attention/blob/main/TropicalAttention.py
class TropicalLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TropicalLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.randn(output_dim, input_dim))
    
    def forward(self, x):
        x_expanded = x.unsqueeze(-2)
        W_expanded = self.W.unsqueeze(0)
        Wx = x_expanded + W_expanded  
        y, _ = torch.max(Wx, dim=-1)
        return y
    
class TropicalAttention(nn.Module):
    def __init__(self, model_dim, num_heads, 
                 activation=F.relu, c_dim=None, gated=False, scaling=False, ffn_factor=4, c_proj=None, ### placeholder to align with std attn
                 tropical_proj=True, tropical_norm=False, symmetric=True):

        super(TropicalAttention, self).__init__()
        assert model_dim % num_heads == 0
        self.d_k = model_dim // num_heads
        self.num_heads = num_heads
        self.tropical_proj = tropical_proj
        self.tropical_norm = tropical_norm
        self.symmetric = symmetric
        
        # Linear layers without bias
        self.out = nn.Linear(model_dim, model_dim, bias=False)

        if self.tropical_proj:
            # Multi-head attention tropical linear map
            self.query_trop = TropicalLinear(self.d_k, self.d_k)
            self.key_trop = TropicalLinear(self.d_k, self.d_k)
            self.value_trop = TropicalLinear(self.d_k, self.d_k)

        if self.tropical_norm:
            self.lambda_param = nn.Parameter(torch.ones(1, 1, model_dim)) #, device=device))

    def tropicalize(self, x):

        # Apply ReLU and log1p in a single pass before linear transformation
        x = torch.log1p(F.relu(x))
        if self.tropical_norm:
            x = x - self.lambda_param
        return x

    def forward(self, x, kv=None, key_padding_mask=None, return_attn_scores=False):
        batch_size, seq_len_q, _ = x.size()
        
        # Determine source for keys/values
        if kv is None:
            kv = x
        seq_len_k = kv.size(1)

        q = self.tropicalize(x)
        k = self.tropicalize(kv)
        v = self.tropicalize(kv)
        
        # Reshape and permute for multi-head attention
        q = q.reshape(batch_size, seq_len_q, self.num_heads, self.d_k).permute(0, 2, 1, 3)  # [B, H, Sq, D]
        k = k.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).permute(0, 2, 1, 3)  # [B, H, Sk, D]
        v = v.reshape(batch_size, seq_len_k, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        
        # Merge batch and heads for parallel computation
        B = batch_size * self.num_heads
        q = q.reshape(B, seq_len_q, self.d_k)  # [B*H, Sq, D]
        k = k.reshape(B, seq_len_k, self.d_k)  # [B*H, Sk, D]
        v = v.reshape(B, seq_len_k, self.d_k)

        # Tropical linear map
        if self.tropical_proj:
            q = self.query_trop(q)
            k = self.key_trop(k)
            v = self.value_trop(v)

        diff = q.unsqueeze(2) - k.unsqueeze(1)   # [B*H, Sq, Sk, D]

        # Compute Hilbert Projective Metric
        if self.symmetric:
            diff = q.unsqueeze(2) - k.unsqueeze(1)  # [B*H, Sq, Sk, D]
            # Calculate tropical distance
            max_diff, _ = diff.max(dim=-1)  # [B*H, Sq, Sk]
            min_diff, _ = diff.min(dim=-1)  # [B*H, Sq, Sk]
            d_trop = max_diff - min_diff    # [B*H, Sq, Sk]
            attn_scores = - d_trop           # Higher scores for closer queries and keys
        else:
            sum_diff = diff.sum(dim=-1)            # [B*H, Sq, Sk]
            min_diff = diff.amin(dim=-1)           # [B*H, Sq, Sk]
            n = q.size(-1)
            attn_scores = - (sum_diff - n * min_diff)
        
        if key_padding_mask is not None:
            mask = key_padding_mask.repeat_interleave(self.num_heads, dim=0).unsqueeze(1)  # [B*H, 1, Sk]
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # Compute context using tropical multiplication and aggregation
        sum_sv = attn_scores.unsqueeze(-1) + v.unsqueeze(1)  # [B*H, Sq, Sk, D]
        context = sum_sv.max(dim=2).values  # [B*H, Sq, D]
        
        # Reshape context back to [batch_size, seq_len, model_dim]
        context = context.reshape(batch_size, self.num_heads, seq_len_q, self.d_k).permute(0, 2, 1, 3).reshape(batch_size, seq_len_q, -1)
        
        # Apply the output linear layer after exponentiation
        context = torch.expm1(context)
        output = self.out(context)
        
        if return_attn_scores:
            return output, attn_scores
        return output