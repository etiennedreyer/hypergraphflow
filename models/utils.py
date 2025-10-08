import torch
from torch.nn.functional import softmax, pad
import numpy as np



def build_1d_sincos_posemb(max_len, embed_dim=1024, temperature=10000.):
    """Sine-cosine positional embeddings from MoCo-v3, adapted back to 1d
    taken from the 4M codebase
    Returns positional embedding of shape (1, N, D)
    """
    arange = torch.arange(max_len, dtype=torch.float32) # Shape (N,)
    assert embed_dim % 2 == 0, 'Embed dimension must be divisible by 2 for 1D sin-cos position embedding'
    pos_dim = embed_dim // 2
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim # Shape (D/2,)
    omega = 1. / (temperature ** omega)
    out = torch.einsum('n,d->nd', [arange, omega]) # Outer product, shape (N, D/2)
    pos_emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1).unsqueeze(0) # Shape (1, N, D)
    return pos_emb


class NeRFEncoding2D(torch.nn.Module):
    def __init__(self, input_dims, num_freqs, include_input=True, output_dim=None):
        """
        "Gemini 2.5 Pro Preview 03-25"
        Implements 2D positional encoding for continuous coordinates as used in NeRF,
        assuming log sampling for frequencies.

        Args:
            input_dims (int): The number of input dimensions (e.g., 2 for (x, y)).
            num_freqs (int): The number of frequency bands (L in the NeRF paper).
                             The total number of sin/cos components per input dim will be 2 * num_freqs.
            include_input (bool): If True, concatenates the original input to the encoding.
            output_dim (int): If provided, aditional liear layer will be added to
        """
        super().__init__()
        if input_dims <= 0:
            raise ValueError("input_dims must be positive.")
        if num_freqs <= 0:
            raise ValueError("num_freqs must be positive for encoding to be meaningful.")
            
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input

        # Frequencies are 2^0, 2^1, ..., 2^(L-1)
        freq_bands = 2.0 ** torch.arange(num_freqs) # Shape (L)
        self.register_buffer('freq_bands_buffer', freq_bands) # Stored as (L)

        self.nerf_output_dims = 0
        if self.include_input:
            self.nerf_output_dims += self.input_dims
        
        # Each input dimension (x, y, ...) gets num_freqs sine and num_freqs cosine components
        self.nerf_output_dims += self.input_dims * self.num_freqs * 2

        if output_dim is not None:
            self.lin_proj = torch.nn.Linear(self.nerf_output_dims, output_dim)
            

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords (torch.Tensor): Input tensor of shape (..., input_dims),
                                   e.g., (N, 2) for N 2D points.
                                   Coordinates are expected to be continuous.
                                   Normalization (e.g. to [-1, 1] or [0, 1]) is often beneficial.
        Returns:
            torch.Tensor: Encoded tensor of shape (..., output_dims).
        """
        if coords.shape[-1] != self.input_dims:
            raise ValueError(f"Expected input last dimension to be {self.input_dims}, but got {coords.shape[-1]}")

        original_shape = coords.shape
        # Flatten all but the last dimension to (B, input_dims) where B is product of leading dims
        coords_flat = coords.reshape(-1, self.input_dims) # (B, D)

        # Prepare for broadcasting:
        # coords_flat.unsqueeze(-1)      # (B, D, 1)
        # self.freq_bands_buffer         # (L) - will broadcast with the last dim of coords_expanded
        
        # scaled_coords will have shape (B, D, L)
        # Each element p_i * 2^k * pi
        scaled_coords = coords_flat.unsqueeze(-1) * self.freq_bands_buffer * math.pi
        
        # Apply sin and cos, each resulting in a tensor of shape (B, D, L)
        sines = torch.sin(scaled_coords)
        cosines = torch.cos(scaled_coords)

        # Concatenate sines and cosines along the last dimension
        # This creates [sin(f0), sin(f1)..., cos(f0), cos(f1)...] for EACH input dimension
        # Resulting shape: (B, D, 2*L)
        encoded_periodic = torch.cat([sines, cosines], dim=-1)

        # Reshape to concatenate the encodings for each dimension sequentially
        # Desired output structure [gamma(p_0), gamma(p_1), ..., gamma(p_D-1)]
        # where gamma(p_d) = [sin(f0_pd), ..., sin(f(L-1)_pd), cos(f0_pd), ..., cos(f(L-1)_pd)]
        # So, we want to flatten the D and 2*L dimensions.
        # The current order in memory after cat is, for each batch item B:
        # dim0: [sin(f0_d0)...sin(fL_d0), cos(f0_d0)...cos(fL_d0)]
        # dim1: [sin(f0_d1)...sin(fL_d1), cos(f0_d1)...cos(fL_d1)]
        # ...
        # Reshaping to (B, D * 2*L) will lay these out sequentially.
        encoded_periodic_flat = encoded_periodic.reshape(coords_flat.shape[0], -1) # (B, D * 2*L)

        if self.include_input:
            final_encoding = torch.cat([coords_flat, encoded_periodic_flat], dim=1) # (B, D + D * 2*L)
        else:
            final_encoding = encoded_periodic_flat # (B, D * 2*L)

        # Reshape back to original batch dimensions plus the new feature dimension
        output_shape = original_shape[:-1] + (self.nerf_output_dims,)
        output = final_encoding.reshape(output_shape)

        if hasattr(self, 'lin_proj'):
            output = self.lin_proj(output)
        
        return output



def attach_context(x, context):
    """Concatenates a context tensor to an input tensor with considerations for
    broadcasting.

    The idea behind this is to allow a context tensor less or equal dimensions to be
    concatenated to an input with more dimensions.
    This function checks the dimension difference and reshapes the context to have
    the same dimension as the constituents so broadcast concatenation can apply.
    The shape change always assumes that the first dimension is the batch and the last
    are the features.

    Here is a basic use case: concatenating the high level jet variables to the
    constituents during a forward pass through the network
    - The jet variables will be of shape: [batch, j_features]
    - The constituents will be of shape: [batch, num_nodes, n_features]
    Here the jet variable will be shaped to [batch, 1, j_features] allowing broadcast
    concatenation with the constituents

    Another example is using the edge features [b,n,n,ef] and concatenating the
    high level jet variables, which will be expanded to [b,1,1,jf] or the conditioning
    on the node features which will be expanded to [b,1,n,nf]
    """
    if context is None:
        raise RuntimeError("Expected context is missing from forward pass")

    # Check if the context information has less dimensions and the broadcast is needed
    if (dim_diff := x.dim() - context.dim()) < 0:
        raise ValueError(
            f"Provided context has more dimensions ({context.dim()}) than inputs ({x.dim()})"
        )

    # If reshaping is required
    if dim_diff > 0:
        # Reshape the context inputs with 1's after the batch dim
        context = add_dims(context, x.dim())

        # Use expand to allow for broadcasting as expand does not allocate memory
        context = context.expand(*x.shape[:-1], -1)

    # Apply the concatenation on the final dimension
    return torch.cat([x, context], dim=-1)


def padded_to_packed(seq, mask):
    # mask: True for valid tokens
    seqlens = mask.sum(dim=-1)
    maxlen = seqlens.max() # .item()
    culens = pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return seq[mask], culens, maxlen


def packed_to_padded(unpadded_seq, mask):
    # mask: True for valid tokens
    shape = (*mask.shape, unpadded_seq.shape[-1])
    out = torch.zeros(shape, dtype=unpadded_seq.dtype, device=unpadded_seq.device)
    out[mask] = unpadded_seq
    return out
