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
        dropout : float,
        format : str = 'batch-time-channel'
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

        self.format = format

        hidden_size = int(input_size * hidden_size_fac)

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(int(hidden_layers)):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.Dropout(p=dropout)
                )
            )

        self.output_layer = nn.Linear(hidden_size, output_size)


    def forward(self, x : torch.Tensor):
        if self.format == 'batch-time-channel':
            x = x.reshape(x.shape[0], -1)  # flatten

        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        if self.format == 'batch-time-channel':
            x = x[:, :, None]

        return x
