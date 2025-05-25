import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    2 layer convolutional neural network with a long-short-term memory network.

    There are two modes available depending on the value of use_pooling:
        - If use_pooling=True, then only the pooled filtered representation is given tp the LSTM.
        - If use_pooling=False, then the filtered representation and original input is given to the LSTM.

    Args:
        input_size (int): The number of channels in the input.
        hidden_size (int): The hidden size of the LSTM.
        output_size (int): The model output size.
        lstm_hidden_layers (int): The number of stacked LSTM layers.
        bidirectional (bool): Make the LSTM bidirectional.
        output_depth (int): If not zero, add a dense output network with this many layers.
        output_size_decay (float): The reduction in layer size per layer of the output network.
        dropout (float): Dropout factor applied after the convolutional layers.
        conv1_filters (int): The number of filters in the first convolutional layer.
        conv2_filters (int): The number of filters in the second convolutional layer.
        use_pooling (bool): Apply max pooling to the filtered representations. See above for more information.
        pool_kernel_size (int): The pools' kernel size.
        pool_stride (int): The pools' stride.
    """
    def __init__(
            self,
            input_size : int,
            hidden_size : int,
            output_size : int,
            lstm_hidden_layers : int,
            bidirectional = False,
            output_depth: int = 4,
            output_size_decay : float = 0.5,
            dropout : float = 0.0,
            conv1_filters: int = 32,
            conv2_filters: int = 64,
            conv_kernel_size: int = 3,
            use_pooling: bool = False,
            pool_kernel_size: int = 2,
            pool_stride: int = 2
        ):
        super(CNNLSTM, self).__init__()

        self.use_pooling = use_pooling

        # kernel size must be odd
        conv_kernel_size = conv_kernel_size-1 if (conv_kernel_size % 2) == 0 else conv_kernel_size

        padding = (conv_kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=conv1_filters,
            kernel_size=conv_kernel_size,
            padding=padding,
            # bias=False  # because we have a batch norm after
        )
        self.pool1 = nn.MaxPool1d(
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            padding=0
        )
        self.bn1 = nn.BatchNorm1d(conv1_filters) # this makes results worse through experiments

        self.conv2 = nn.Conv1d(
            in_channels=conv1_filters,
            out_channels=conv2_filters,
            kernel_size=conv_kernel_size,
            padding=padding,
            # bias=False  # because we have a batch norm after
        )
        self.bn2 = nn.BatchNorm1d(conv2_filters) # this makes results worse through experiments
        self.pool2 = nn.MaxPool1d(
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            padding=0
        )

        self.relu = nn.LeakyReLU()

        if self.use_pooling:
            lstm_input_size = conv2_filters
        else:
            lstm_input_size = conv2_filters + input_size
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=lstm_hidden_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(p=dropout)

        self.output_layers = nn.ModuleList()
        for i in range(int(output_depth)):
            in_features = int(hidden_size * output_size_decay**i)
            out_features = int(hidden_size * output_size_decay**(i+1))
            self.output_layers.append(nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.LayerNorm(out_features),
                nn.LeakyReLU()
            ))
        self.last_fc = nn.Linear(int(hidden_size * output_size_decay**output_depth), output_size)

    def forward(
            self,
            x : torch.Tensor | nn.utils.rnn.PackedSequence,
            h = None,
            c = None,
            state_connected=False,
            packed_input=False):
        
        if packed_input:
            x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Transpose for Conv1d: (batch_size, seq_len, input_size) → (batch_size, input_size, seq_len)
        # i.e. convolutions in torch require shape=(batch_size, in_channels, seq_len)
        x_conv = x.transpose(1, 2)  # Now (batch_size, input_size, seq_len)

        # Apply convolutions
        x_conv = self.conv1(x_conv)  # (batch_size, conv_filters, reduced_seq_len)
        # x_conv = self.bn1(x_conv)  # this makes results worse through experiments
        if self.use_pooling:
            x_conv = self.pool1(x_conv)
        x_conv = self.relu(x_conv)
        x_conv = self.dropout(x_conv)
        x_conv = self.conv2(x_conv)
        # x_conv = self.bn2(x_conv) # this makes results worse through experiments
        if self.use_pooling:
            x_conv = self.pool2(x_conv)
        x_conv = self.relu(x_conv)
        x_conv = self.dropout(x_conv)
        # Transpose back for LSTM: (batch_size, conv_filters, seq_len) → (batch_size, seq_len, conv_filters)
        x_conv = x_conv.transpose(1, 2)

        if self.use_pooling:
            x = x_conv
        else:
            x_combined = torch.cat((x, x_conv), dim=2)  # concatenate the convolution features with the original input
            x = x_combined

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
        for layer in self.output_layers:
            out = layer(out)
        out = self.last_fc(out)

        # unsqueeze creates time dimension so we can compare to train data easily
        if state_connected:
            return out.unsqueeze(1).transpose(1, 2), (h, c)
        else:
            return out.unsqueeze(1).transpose(1, 2)
