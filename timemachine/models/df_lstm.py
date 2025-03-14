import torch
import torch.nn as nn
from timemachine.models.feature_layer.differentiable_feature_layer import DifferentiableFeatureLayer



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
            output_depth : int = 4,
            output_size_decay : float = 0.5,
            dropout : float = 0,
        ):
        
        super(DFLSTM, self).__init__()

        self.df_layer = DifferentiableFeatureLayer(
            feature_configs=feature_config,
            input_dim=input_size,
            pad_mode='zeros'
        )

        self.lstm = nn.LSTM(
            input_size=self.df_layer.output_dim,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True)
        
        self.dropout = nn.Dropout(p=dropout)

        self.posterior_layers = nn.ModuleList()
        for i in range(int(output_depth)):
            in_features = int(hidden_size * output_size_decay**i)
            out_features = int(hidden_size * output_size_decay**(i+1))
            self.posterior_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features),
                nn.LeakyReLU()
            ))
        self.last_fc = nn.Linear(int(hidden_size * output_size_decay**output_depth), output_size)
    
    def forward(
            self,
            x : torch.Tensor | nn.utils.rnn.PackedSequence,
            full_buffer : torch.Tensor,
            indices : torch.Tensor,
            h = None,
            c = None,
            state_connected=True,
            packed_input=False):
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

        if state_connected and (h is None or c is None):
            batch_size = x.batch_sizes[0]  # Get the batch size from packed sequence
            h = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.data.device)
            c = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.data.device)

        # this probably wont work for now, keep packing off
        if packed_input:
            packed_out, (h, c) = self.lstm(x, (h, c) if h is not None else None)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, (h, c) = self.lstm(x, (h, c) if h is not None else None)

        out = h[-1]
        for layer in self.posterior_layers:
            out = layer(out)
        out = self.last_fc(out)

        # unsqueeze simply create a time dimension so we can compare to train data easily
        if state_connected:
            return out.unsqueeze(1), (h, c)
        else:
            return out.unsqueeze(1), (None, None)
