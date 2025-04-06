import torch
import torch.nn as nn



class StatefulLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            layers,
            output_depth:int=4,
            output_size_decay=0.5,
            dropout=0,
            bidirectional:bool=False
        ):
        
        super(StatefulLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(p=dropout)

        self.posterior_layers = nn.ModuleList()
        for i in range(int(output_depth)):
            in_features = int(hidden_size * output_size_decay**i)
            out_features = int(hidden_size * output_size_decay**(i+1))
            self.posterior_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features),
                nn.ReLU()
            ))
        self.last_fc = nn.Linear(int(hidden_size * output_size_decay**output_depth), output_size)

    def forward(
            self,
            x : torch.Tensor | nn.utils.rnn.PackedSequence,
            h = None,
            c = None,
            state_connected=True,
            packed_input=True):
        if state_connected and (h is None or c is None):
            batch_size = x.batch_sizes[0]  # Get the batch size from packed sequence
            h = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.data.device)
            c = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.data.device)
        if packed_input:
            packed_out, (h, c) = self.lstm(x, (h, c) if h is not None else None)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, (h, c) = self.lstm(x, (h, c) if h is not None else None)

        out = h[-1]
        for layer in self.posterior_layers:
            out = layer(out)
        out = self.last_fc(out)

        # unsqueeze creates a singleton time dimension NOTE this could be removed
        if state_connected:
            return out.unsqueeze(1), (h, c)
        else:
            return out.unsqueeze(1), (None, None)
