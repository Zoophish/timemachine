import torch
import torch.nn as nn
from timemachine.models.feature_layer.differentiable_feature_layer import DifferentiableFeatureLayer
from timemachine.models.stateful_lstm import StatefulLSTM
from timemachine.models.mlp import HourglassMLP



class DFLSTM(nn.Module):
    """
    LSTM with a differentiable feature layer on the front.
    """
    def __init__(
            self,
            input_size : int,
            hidden_size : int,
            output_size : int,
            lstm_layers : int,
            feature_config : list,
            device : torch.device,
            dropout : float = 0,
            bidirectional : bool = False,
            pool : str = 'max'
        ):
        
        super(DFLSTM, self).__init__()

        self.df_layer = DifferentiableFeatureLayer(
            feature_configs=feature_config,
            input_dim=input_size,
            pad_mode='zeros',
            device = device
        )

        self.lstm = StatefulLSTM(
            input_size=self.df_layer.output_dim,
            output_size=output_size,
            hidden_size=hidden_size,
            layers=lstm_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            pool=pool
        )
            
    def forward(
            self,
            x : torch.Tensor,
            full_buffer : torch.Tensor,
            indices : torch.Tensor,
        ):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            full_buffer: Full time series data of shape (series_len, input_dim)
            indices: Tensor of shape (batch_size, 2) containing the start and end indices of x in full_series
            h: Hidden state,
            c: Cell state,
            state_connected: Use the provided hidden/cell states and provide the new ones
            packed_input: x is a packed sequence
        """
        
        x = self.df_layer(x, full_buffer, indices)
        x = self.lstm(x)
        return x


class DFDNN(nn.Module):
    """
    MLP with a differentiable feature layer on the front.
    """
    def __init__(
            self,
            input_size : int,
            window_size : int,
            hidden_size_fac : float,
            output_size : int,
            hidden_layers : int,
            dropout : float,
            feature_config : list,
            device : torch.device,
        ):
        
        super(DFDNN, self).__init__()

        self.window_size = window_size

        self.df_layer = DifferentiableFeatureLayer(
            feature_configs=feature_config,
            input_dim=input_size,
            pad_mode='zeros',
            device = device
        )

        self.mlp = HourglassMLP(
            input_size=self.df_layer.output_dim * window_size,
            hidden_size_fac=hidden_size_fac,
            hidden_layers=hidden_layers,
            output_size=output_size,
            dropout=dropout,
            format='null'
        )
    
    def forward(
        self,
        x : torch.Tensor,
        full_buffer : torch.Tensor,
        indices : torch.Tensor,
    ):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            full_buffer: Full time series data of shape (series_len, input_dim)
            indices: Tensor of shape (batch_size, 2) containing the start and end indices of x in full_series
            h: Hidden state,
            c: Cell state,
            state_connected: Use the provided hidden/cell states and provide the new ones
            packed_input: x is a packed sequence
        """
        assert x.shape[1] == self.window_size, f"Input sequence is not of window size ({self.window_size})"
        
        x = self.df_layer(x, full_buffer, indices)  # generate features
        x = x.reshape(x.shape[0], self.df_layer.output_dim * self.window_size)  # flatten
        x = self.mlp(x)
        return x[:, :, None]
