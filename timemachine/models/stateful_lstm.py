import torch
import torch.nn as nn



class StatefulLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            layers,
            dropout = 0,
            bidirectional : bool = False,
            pool: str = 'max',
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

        self.pool = pool
        if pool == 'attention':
            self.attention_layer = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
            
        hidden_size = 2 * hidden_size if bidirectional else hidden_size
        self.last_fc = nn.Linear(hidden_size, output_size)

    def forward(
            self,
            x : torch.Tensor | nn.utils.rnn.PackedSequence,
            h = None,
            c = None,
            state_connected=False,
            packed_input=False
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

        if self.pool == 'mean':
            out = out.mean(dim=1)
        elif self.pool == 'max':
            out = out.max(dim=1).values
        elif self.pool == 'attention':
            attn_scores = self.attention_layer(out).squeeze(-1)  # (batch_size, seq_len)
            attn_weights = torch.softmax(attn_scores, dim=1)     # (batch_size, seq_len)    
            out = torch.bmm(attn_weights.unsqueeze(1), out).squeeze(1)
        else:
            out = out[:, -1, :]

        out = self.last_fc(out)

        if state_connected:
            return out.unsqueeze(1).transpose(1, 2), (h, c)
        else:
            return out.unsqueeze(1).transpose(1, 2)
