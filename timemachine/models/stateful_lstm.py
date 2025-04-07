import torch
import torch.nn as nn



class StatefulLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            layers,
            dropout=0,
            bidirectional:bool=False
        ):
        
        super(StatefulLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)

        hidden_size = 2 * hidden_size if bidirectional else hidden_size
        self.last_fc = nn.Linear(hidden_size, output_size)

    def forward(
            self,
            x : torch.Tensor | nn.utils.rnn.PackedSequence,
            h = None,
            c = None,
            state_connected=True,
            packed_input=True
        ):
        if state_connected and (h is None or c is None):
            batch_size = x.batch_sizes[0]  # Get the batch size from packed sequence
            num_directions = 2 if self.lstm.bidirectional else 1
            h = torch.zeros(self.lstm.num_layers * num_directions, batch_size, self.lstm.hidden_size).to(x.data.device)
            c = torch.zeros(self.lstm.num_layers * num_directions, batch_size, self.lstm.hidden_size).to(x.data.device)

        if packed_input:
            packed_out, (h, c) = self.lstm(x, (h, c) if h is not None else None)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, (h, c) = self.lstm(x, (h, c) if h is not None else None)

        if self.lstm.bidirectional:
            out = torch.cat([h[-2], h[-1]], dim=1)
        else:
            out = h[-1]

        out = self.last_fc(out)

        # unsqueeze creates a singleton time dimension NOTE this could be removed
        if state_connected:
            return out.unsqueeze(1), (h, c)
        else:
            return out.unsqueeze(1), (None, None)
