import torch.nn as nn


class ConfigurableEmbedding(nn.Module):
    """MLP-based embedding layer with configurable depth."""

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        layers = []
        current_dim = input_dim

        if hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dims[0]))
            layers.append(nn.ReLU())
            current_dim = hidden_dims[0]

            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(current_dim, hidden_dims[i + 1]))
                layers.append(nn.ReLU())
                current_dim = hidden_dims[i + 1]

        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
