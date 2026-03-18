import torch
import torch.nn as nn

class RandomFourierEmbedder(nn.Module):

    def __init__(self,input_dim: int, output_dim: int, scale: float = 10):

        super().__init__()
        assert output_dim % 2 == 0, "output_dim must be even"
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim//2) * scale,
                                    requires_grad=False)
    
    def forward(self, x):

        x = x @ self.weights
        s = torch.sin(2*torch.pi*x)
        c = torch.cos(2*torch.pi*x)
        return torch.cat([s, c], dim=-1)