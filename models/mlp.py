import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, layers, output_dim, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = self.get_activation(activation)
        self.layers.append(nn.Linear(input_dim, layers[0]))
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(self.activation)
        self.layers.append(nn.Linear(layers[-1], output_dim))

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'silu':
            return nn.SiLU()
        else:
            raise NotImplementedError(f"Please implement {activation}")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x