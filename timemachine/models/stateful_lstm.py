import torch
import torch.nn as nn



class StatefulLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers, posterior_depth:int=4, posterior_decay=0.5, dropout=0):
        super(StatefulLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)

        # self.fc = nn.Linear(hidden_size, output_size)

        self.posterior_layers = nn.ModuleList()
        for i in range(int(posterior_depth)):
            in_features = int(hidden_size * posterior_decay**i)
            out_features = int(hidden_size * posterior_decay**(i+1))
            self.posterior_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features),
                nn.ReLU()
            ))
        self.last_fc = nn.Linear(int(hidden_size * posterior_decay**posterior_depth), output_size)

        # self.fc1 = nn.Linear(hidden_size, int(hidden_size*bottleneck))
        # self.fc2 = nn.Linear(int(hidden_size*bottleneck), output_size)
        # self.relu = nn.ReLU()
    
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
        
        # out = self.dropout(out)
        # out = self.fc(h[-1])

        # out = self.dropout(out[:, -1, :])
        # out = self.relu(self.fc1(out))
        # out = self.fc2(out)

        out = h[-1]
        for layer in self.posterior_layers:
            out = layer(out)
        out = self.last_fc(out)

        # unsqueeze simply create a time dimension so we can compare to train data easily
        if state_connected:
            return out.unsqueeze(1), (h, c)
        else:
            return out.unsqueeze(1), (None, None)
