import torch
import torch.nn as nn


class HourglassMLP(nn.Module):
    """
    Multilayer perceptron (feed-forward neural network) with reduced size hidden layers
    (like an hourglass or autoencoder network).
    """
    def __init__(
        self,
        input_size : int,
        hidden_size_fac : float,
        hidden_layers : int,
        output_size : int,
        dropout : float
    ):
        """
        Args:
            input_size (int): Input layer size
            hidden_size_fac (float): Hidden layer(s) size as a fraction of the input size
            hidden_layers (int): Number of hidden layers
            output_size (int): Output layer size
            dropout (float): Hidden layer dropout factor
        """
        super(HourglassMLP, self).__init__()

        hidden_size = int(input_size * hidden_size_fac)

        self.input_layer = nn.Linear(input_size, hidden_size)

        self.hidden_layers = nn.ModuleList()
        for i in range(int(hidden_layers)):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(p=dropout)
                )
            )

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x : torch.Tensor):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
