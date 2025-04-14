import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_fn):
        """
        Initializes the Weighted MSE Loss.
        :param weight_fn: A function that generates weights for each timestep in the series.
                          The function should take the series length as input.
        """
        super(WeightedMSELoss, self).__init__()
        self.weight_fn = weight_fn

    def forward(self, predictions, targets):
        """
        Computes the weighted MSE loss.
        :param predictions: Model predictions (shape: [batch_size, seq_len])
        :param targets: Ground truth values (shape: [batch_size, seq_len])
        :return: Weighted MSE loss.
        """
        seq_len = predictions.size(1)
        weights = self.weight_fn(seq_len)  # Shape: [seq_len]
        weights = weights.to(predictions.device)  # Move weights to the correct device

        # Reshape weights for broadcasting: [1, seq_len, 1]
        weights = weights.view(1, seq_len, 1)
        
        # Compute weighted MSE
        squared_diff = (predictions - targets) ** 2
        weighted_loss = weights * squared_diff
        return torch.mean(weighted_loss)

def linear_weights(seq_len):
    return torch.linspace(0.1, 1.0, steps=seq_len)

def exponential_weights(seq_len):
    return torch.exp(torch.linspace(0, 1, steps=seq_len))

# this boosts the first point to reduce compound error, then applies a gradually increasing weighting
def first_exponential_weights(seq_len):
    if seq_len <= 1:
        return torch.tensor([2.71])
    first_weight = torch.tensor([1.0])
    exp_weights = torch.exp(torch.linspace(0, 1, steps=seq_len-1))
    return torch.cat([first_weight, exp_weights])



class PenalisedSignLoss(nn.Module):
    def __init__(self, scaler, penalty_factor=10.0):
        """
        Custom loss that penalises wrong-sign predictions heavily.
        It assumes the sign is not preserved from the unscaled data; as such, the scaler must be provided.
        
        Args:
            scaler (sklearn scaler): Scaler used to normalise targets, needed for inverse-scaling.
            penalty_factor (float): Factor to penalise wrong-sign predictions.
        """
        super(PenalisedSignLoss, self).__init__()
        self.scaler = scaler
        self.penalty_factor = penalty_factor
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred_scaled, y_true_scaled):
        # Inverse scale outputs
        y_pred = torch.tensor(self.scaler.inverse_transform(y_pred_scaled.detach().cpu().numpy()), requires_grad=True)
        y_true = torch.tensor(self.scaler.inverse_transform(y_true_scaled.detach().cpu().numpy()), requires_grad=False)
        
        # Convert to tensors for computation
        y_pred = y_pred.to(y_pred_scaled.device)
        y_true = y_true.to(y_pred_scaled.device)

        # MSE loss
        mse = self.mse_loss(y_pred_scaled, y_true_scaled)
        
        # Wrong-sign penalty
        sign_mismatch = (torch.sign(y_pred) != torch.sign(y_true)).float()
        sign_loss = self.penalty_factor * sign_mismatch * torch.abs(y_true - y_pred)
        
        # Combine losses
        total_loss = mse + sign_loss.mean()
        return total_loss



def divide_no_nan(a, b):
    div = a / b
    div[div != div] = 0.0
    div[div == float('inf')] = 0.0
    return div


class MAPELoss(nn.Module):
    """
    Mean absolute percentage error.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        percentage_errors = torch.abs((targets - predictions) * divide_no_nan(torch.ones_like(targets),  torch.abs(targets)))
        
        mape = 100.0 * torch.mean(percentage_errors)
        
        return mape
    
class MAELoss(nn.Module):
    """
    Mean absolute error loss.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        mae = torch.mean(torch.abs(predictions - targets))
        return mae
    

class QuantileLoss(nn.Module):
    """
    Quantile loss for multiple quantiles across time, computed efficiently using tensor operations.
    Input shape: predictions [batch_size, time, num_quantiles], targets [batch_size, time]
    """
    def __init__(self, quantiles, epsilon=1e-8):
        super().__init__()
        assert all(0 < q < 1 for q in quantiles), "Quantiles must be between 0 and 1"
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)  # e.g., [0.1, 0.5, 0.9]
        self.epsilon = epsilon
        
    def forward(self, predictions, targets):
        # predictions shape: [batch_size, time, num_quantiles]
        # targets shape: [batch_size, time] or [batch_size, time, 1]
        assert predictions.dim() == 3, "Predictions must have shape [batch_size, time, num_quantiles]"
        assert predictions.shape[2] == len(self.quantiles), "Number of quantiles in predictions must match quantiles provided"
        
        # Ensure targets has an extra dimension for broadcasting
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)  # [batch_size, time, 1]
        
        assert targets.shape[:2] == predictions.shape[:2], "Batch and time dimensions must match between predictions and targets"
        
        # Move quantiles to the device of predictions and add dimensions for broadcasting
        quantiles = self.quantiles.to(predictions.device).view(1, 1, -1)  # [1, 1, num_quantiles]
        
        # Compute errors: [batch_size, time, num_quantiles]
        errors = targets - predictions
        
        # Compute quantile loss using broadcasting
        losses = torch.where(
            errors >= 0,
            quantiles * errors,          # Underprediction: q * error
            (1 - quantiles) * -errors    # Overprediction: (1 - q) * (-error)
        )
        
        # Mean over all dimensions (batch, time, quantiles)
        total_loss = torch.mean(losses)
        # Alternative: torch.mean(losses, dim=[0, 1]).sum() to sum per-quantile losses over batch and time
        
        return total_loss