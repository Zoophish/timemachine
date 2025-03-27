import torch
import torch.nn as nn
from typing import List, Dict, Callable, Tuple, Optional, Union



class DifferentiableFeatureLayer(nn.Module):
    """
    A PyTorch layer that applies feature transformations to time series data.
    You can use arbitrary feature transformations, as long as the parameters of the transformation
    are differentiable (i.e. don't break the compute graph of the tensors).

    The parameters of these transformations are then learnable through backpropagation when the layer is coupled
    to a forecasting model and loss function.
    
    This layer requires direct accesss to the full time series to dynamically compute rolling features.
    """
    
    def __init__(
        self,
        feature_configs: List[Dict],
        input_dim: int,
        device : torch.device,
        max_lookback_size : int = 128,
        pad_mode : str = 'zeros',
    ):
        """
        Initialize the differentiable feature layer.
        
        Args:
            feature_configs: List of feature configuration dictionaries.
                Each dict should contain:
                - 'name': Feature name (str)
                - 'function': Feature function to apply
                - 'learnable_params': Dict of parameter names and initial values
                - 'fixed_params': Dict of fixed parameter values (optional)
                - 'out_size': int, optional. The output dimensions of the feature. 1 if unspecified.
            input_dim: Number of input features in the time series
            max_lookback_size: A limit on how many previous time steps the feature funcs can get per batch.
            pad_mode: How to handle padding for lookback windows ('zeros' or 'replicate')
        """
        super().__init__()
        
        self.feature_configs = feature_configs
        self.input_dim = input_dim
        self.pad_mode = pad_mode
        self.device = device

        self.batch_norms = nn.ModuleList()
        for _ in range(len(feature_configs)):
            self.batch_norms.append(nn.BatchNorm1d(input_dim))

        self.max_lookback_size = max_lookback_size

        # determine total feature output dimensions
        self.feature_output_dims = sum([config.get('out_size', 1) for config in self.feature_configs])
        
        self.output_dim = input_dim * (1 + self.feature_output_dims)
        
        self.feature_info = {}
        self.learnable_params = nn.ParameterDict()
        
        self.fixed_params = {}
        
        for feature_idx, config in enumerate(feature_configs):
            feature_name = config['name']
            
            # register learnable parameters
            for param_name, param_config in config.get('learnable_params', {}).items():
                param_key = f"{feature_name}_{param_name}"
                self.feature_info[param_key] = {}
                
                # check if param_config is a range or a single value
                if isinstance(param_config, (list, tuple)) and len(param_config) == 2:
                    min_val, max_val = param_config
                    self.feature_info[param_key]['min'] = min_val
                    self.feature_info[param_key]['max'] = max_val

                    init_value = min_val + (max_val - min_val) * torch.rand((self.input_dim, ))
                    
                    # for parameters that will use sigmoid, convert to logit space
                    if param_name in ['alpha', 'decay', 'threshold', 'window_size']:
                        target = (init_value - min_val) / (max_val - min_val)  # normalize to 0-1
                        init_value = torch.log(target / (1.0 - target))
                else:
                    assert False, "Not implemented yet"
                    init_value = param_config
                
                # create the parameter vector (for each input channel)
                self.learnable_params[param_key] = nn.Parameter(init_value)
            
            # store fixed parameters
            for param_name, value in config.get('fixed_params', {}).items():
                param_key = f"{feature_name}_{param_name}"
                self.fixed_params[param_key] = torch.full((self.input_dim, ), value)
    
    def _get_param(self, feature_name: str, param_name: str) -> Union[torch.Tensor, float, int]:
        """Get a parameter (learnable or fixed) for a feature."""
        param_key = f"{feature_name}_{param_name}"
        
        if param_key in self.learnable_params:
            # Apply constraints to learnable parameters
            raw_param = self.learnable_params[param_key]

            min_val = self.feature_info[param_key]['min']
            max_val = self.feature_info[param_key]['max']

            # scale to get a value between 1 and max_window
            return min_val + torch.sigmoid(raw_param) * (max_val - min_val)
        
        elif param_key in self.fixed_params:
            return self.fixed_params[param_key]
        
        else:
            raise KeyError(f"Parameter {param_key} not found")
        

    def print_params(self, show_gradients : bool=False):
        for name, param in self.learnable_params.items():
            if 'window_size' in name:
                window_size = torch.floor(1.0 + torch.sigmoid(param) * 100)
                print(f"  {name}: {str(window_size)}")
                if show_gradients:
                    if param.grad is not None:
                        print(f"  {name} grad: {str(param.grad)}")
                    else:
                        print(f"  {name} no gradient!")
            elif any(term in name for term in ['alpha', 'decay', 'threshold']):
                value = torch.sigmoid(param)
                print(f"  {name}: {str(value)}")
                if show_gradients:
                    if param.grad is not None:
                        print(f"  {name} grad: {str(param.grad)}")
                    else:
                        print(f"  {name} no gradient!")
            else:
                print(f"  {name}: {str(param)}")
                if show_gradients:
                    if param.grad is not None:
                        print(f"  {name} grad: {str(param.grad)}")
                    else:
                        print(f"  {name} no gradient!")
    

    def _get_batch_lookback(
            self,
            indices : torch.Tensor,
            full_series : torch.Tensor
        ) -> torch.Tensor:
        """
        For each local window in the batch, fetches the max_lookback_size elements from the full series
        that precede that window. This is so feature operations can access elements further back
        than the window, but without loading the entire full series every time.

        Applies padding if the lookback goes beyond the beginning of full series.

        Args:
            indices: The start and end indices in the full series for each batch item. (batch_size, 2)
            full_series: The entire time series data. (length, input_dim)
        Returns:
            torch.Tensor of shape (batch_size, max_lookback_size, input_dim).
        """
        batch_size, _ = indices.shape
        _, input_dim = full_series.shape
        seq_len = indices[0, 1].item() - indices[0, 0].item()

        lookback_batch = torch.zeros(  # zeros padding for now
            (batch_size, self.max_lookback_size + seq_len, input_dim),
            device=self.device
        )
        
        for b in range(batch_size):
            lookback = self.max_lookback_size
            # start/end index for this batch item's data in full_series
            seq_start_idx = indices[b, 0].item()
            seq_end_idx = indices[b, 1].item()
            
            # get the global start index of the lookback window for this batch
            window_start_idx = max(0, seq_start_idx - lookback)

            # account for padding (lookback window goes past the start of full_series, where there's no data - leave as 'zeros')
            # we could have pad the data retrieved from full_series below, but this should me marginally faster
            pad_size = abs(min(seq_start_idx - lookback, 0))

            # the sequence goes back enough for the window start for the first element and the end of the input sequence
            lookback_batch[b, pad_size:, :] = full_series[window_start_idx:seq_end_idx]
        return lookback_batch
    

    def _apply_feature(
        self,
        feature_config: Dict,
        x: torch.Tensor,
        full_series: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies a feature transformation to all channels in the input tensor,
        vectorized across the batch dimension.
        
        Args:
            feature_config: Feature configuration dictionary
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            full_series: Full time series data of shape (series_len, input_dim)
            indices: Tensor of shape (batch_size, 2) containing the start and end indices of x in full_series
            
        Returns:
            Transformed tensor of shape (batch_size, seq_len, input_dim)
        """

        feature_name = feature_config['name']
        feature_func = feature_config['function']
        feature_type = feature_config['type']
        feature_output_size = feature_config.get('out_size', 1) * self.input_dim
        batch_size, window_size, input_dim = x.shape
        device = x.device
        
        # get the learnable parameters
        params = {
            param_name: self._get_param(feature_name, param_name)
            for param_name in feature_config.get('learnable_params', {}).keys()
        }
        # get the fixed parameters
        for param_name in feature_config.get('fixed_params', {}).keys():
            params[param_name] = self._get_param(feature_name, param_name)
        
        # get start and end indices for each batch item
        global_start_indices = indices[:, 0]
        global_end_indices = indices[:, 1]
        assert torch.all(global_end_indices - global_start_indices == window_size), "Window size must be the same across the batch"
        
        # output tensor (batch_size, seq_len, feature_output_size)
        result = torch.zeros((batch_size, window_size, feature_output_size), device=device)

        if feature_type == 'rolling':
            # precompute extra lookback for the batch
            lookback_batch = self._get_batch_lookback(
                indices=indices,
                full_series=full_series
            )

            # batch_windows contains max_window size, but we need to 'slide' a view over this for each element in the sequence
            # Notice that we slide the start of the window here with t, but because each batch item may have a different window size,
            # the end of the window needs to be handled in the feature function
            for t in range(window_size):  # apply the feature func for each position in the input sequence
                # the centre is counterintuitively the front of the window (formally because it's symmetric)
                centre = self.max_lookback_size + t
                result[:, t, :] = feature_func(lookback_batch[:, t:, :], max_window=self.max_lookback_size, **params)

        elif feature_type == 'recurrent':
            batch_windows = self._get_batch_lookback(
                indices=indices,
                full_series=full_series
            )

            # feature doesn't use a window - use the input tensor x
            # (confusion note) instead of sliding the back of the window, we slide the front
            context = torch.zeros((batch_size, 1, input_dim), device=device)
            for t in range(0, window_size): # this is very slow, should consider changing it
                result[:, t, :] = feature_func(batch_windows[:, 0:t+1, :], context, **params)

        return result
    

    def forward(
        self,
        x: torch.Tensor,
        full_series: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the differentiable feature layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            full_series: Full time series data of shape (series_len, input_dim)
            indices: Tensor of shape (batch_size, 2) containing the start and end indices of x in full_series
            
        Returns:
            Transformed tensor with original features and engineered features
            Shape: (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # initialize output tensor with the original features
        output = torch.zeros((batch_size, seq_len, self.output_dim), device=x.device)
        output[:, :, :input_dim] = x
        
        # apply each feature transformation
        for i, feature_config in enumerate(self.feature_configs):
            feature_output = self._apply_feature(feature_config, x, full_series, indices)

            # apply batch norm - need to transpose because BatchNorm1d operates on dim 1
            feature_output = feature_output.transpose(1, 2)
            feature_output = self.batch_norms[i](feature_output)
            feature_output = feature_output.transpose(1, 2)

            output_size = feature_config.get('out_size', 1)
            output_start_idx = input_dim * (i + 1)
            output_end_idx = output_start_idx + input_dim * output_size
            output[:, :, output_start_idx:output_end_idx] = feature_output
        
        return output


def indices_to_sample(
        indices : torch.Tensor,
        full_series: torch.Tensor,
        scaler = None
    ) -> torch.tensor:
    """
    Converts the batched indices to a batch of tensors from the full series.

    Args:
        indices: Tensor of integer indices of shape (batch_size, 2)
        full_series: The full time series data (total_len, input_dim)
        scaler: Input scaler, optional.
    Returns:
        
    """

    batch_size, _ = indices.shape
    seq_len = indices[0, 1].item() - indices[0, 0].item() # all batch items should be the same length
    series_len, num_channels = full_series.shape

    out = torch.zeros((batch_size, seq_len, num_channels), dtype=torch.float32)
    for b in range(batch_size):
        index0 = indices[b, 0].item()
        index1 = indices[b, 1].item()
        if scaler is not None:
            out[b, :, :] = torch.tensor(
                scaler.transform(full_series[index0 : index1, :]),
                dtype=torch.float32,
            )
        else:
            out[b, :, :] = full_series[index0 : index1, :]
    
    return out