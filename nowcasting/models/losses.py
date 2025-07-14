import torch
import torch.nn as nn
from torch.nn import functional as F

class BalancedLoss(nn.Module):
    def __init__(self, min_val=0, max_val=55, max_weight_r=10.0, weight_intensity=1.0, extended=False):
        """
        Initializes the BalancedLoss.
        Args:
            min_val (float): Minimum dBZ value (corresponding to normalized 0). Default: 0.
            max_val (float): Maximum dBZ value (corresponding to normalized 1). Default: 55.
            min_weight (float): Minimum rain rate (mm/h) for clamping the raw weights. Default: 1.0.
            max_weight (float): Maximum rain rate (mm/h) for clamping the raw weights. Default: 10.0.
            weight_intensity (float): A scalar to control the influence of the calculated weights.
                                      0.0 means pure MSE, 1.0 means full weighting. Default: 1.0.
        """
        super(BalancedLoss, self).__init__()
        self.register_buffer("min_val_dbz", torch.tensor(float(min_val), dtype=torch.float32))
        self.register_buffer("max_val_dbz", torch.tensor(float(max_val), dtype=torch.float32))
        self.register_buffer("max_weight_r", torch.tensor(float(max_weight_r), dtype=torch.float32))
        self.register_buffer("weight_intensity", torch.tensor(float(weight_intensity), dtype=torch.float32))
        self.extended = extended

    def forward(self, pred_normalized_dbz, target_normalized_dbz):
        """
        Computes the weighted Mean Squared Error loss.
        Weights are derived from the ground truth target's rain rate, and their influence
        is scaled by `weight_intensity`.

        Args:
            pred_normalized_dbz (torch.Tensor): Model's prediction, normalized to [0,1] dBZ.
            target_normalized_dbz (torch.Tensor): Ground truth target, normalized to [0,1] dBZ.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Denormalize target from [0,1] back to dBZ space
        range_dbz = self.max_val_dbz - self.min_val_dbz
        target_dBZ = target_normalized_dbz * range_dbz + self.min_val_dbz

        # 2. Convert target dBZ to Rain Rate (R) for calculating raw weights
        power_target = target_dBZ / 10.0
        
        base_for_r_calc = (torch.pow(10.0, power_target) - 1.0) / 200.0
        r_from_target = torch.pow(torch.relu(base_for_r_calc), 5.0 / 8.0)

        # 3. Compute the raw weights based on the GROUND TRUTH rain rate
        raw_weights = torch.clamp(r_from_target, max=self.max_weight_r) + 1

        # 4. Apply the weight_intensity to create the effective weights
        # This is the new part:
        # If weight_intensity = 0, effective_weights = 1.0 + (raw_weights - 1.0) * 0 = 1.0
        # If weight_intensity = 1, effective_weights = 1.0 + (raw_weights - 1.0) * 1 = raw_weights
        effective_weights = 1.0 + (raw_weights - 1) * self.weight_intensity

        # 5. Compute the squared error in the normalized dBZ space
        mse = (pred_normalized_dbz - target_normalized_dbz) ** 2

        # 6. Compute weighted MSE with the effective weights
        weighted_mse = effective_weights * mse
        
        if self.extended:
            mae = torch.abs(pred_normalized_dbz - target_normalized_dbz)
            weighted_mae = effective_weights * mae
            return torch.mean(weighted_mse) + torch.mean(weighted_mae)

        return weighted_mse.mean()
    

class MultiSourceLoss(nn.Module):
    def __init__(self, 
                 balanced_channels, 
                 mse_channels, 
                 channel_wegihts,
                 min_val=0, 
                 max_val=55, 
                 max_weight_r=10.0, 
                 weight_intensity=1.0, 
                 extended=False):
        """
        Args:
            balanced_channels (list of int): Indices of channels to apply BalancedLoss.
            mse_channels (list of int): Indices of channels to apply plain MSE loss.
            weights (list of float): Weight for each channel (same length as balanced + mse).
            Other args are forwarded to BalancedLoss.
        """
        super(MultiSourceLoss, self).__init__()

        assert len(channel_wegihts) == len(balanced_channels) + len(mse_channels), \
            "weights length must match total number of channels"

        self.balanced_channels = balanced_channels
        self.mse_channels = mse_channels
        self.channel_wegihts = channel_wegihts

        self.balanced_loss = BalancedLoss(
            min_val=min_val,
            max_val=max_val,
            max_weight_r=max_weight_r,
            weight_intensity=weight_intensity,
            extended=extended
        )

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): (B, T, H, W, C) prediction tensor
            target (Tensor): (B, T, H, W, C) target tensor
        Returns:
            Scalar tensor representing the weighted total loss.
        """
        total_loss = 0.0
        weight_idx = 0

        for c in self.balanced_channels:
            loss = self.balanced_loss(pred[..., c], target[..., c])
            total_loss += self.channel_wegihts[weight_idx] * loss
            weight_idx += 1

        for c in self.mse_channels:
            mse_loss = F.mse_loss(pred[..., c], target[..., c])
            total_loss += self.channel_wegihts[weight_idx] * mse_loss
            weight_idx += 1

        return total_loss
