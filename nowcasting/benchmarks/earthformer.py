import torch
import numpy as np

from nowcasting.models.earthformer.cuboid_transformer import CuboidTransformerModel
from nowcasting.utils import data_prep

class EarthformerModel:
    """
    A wrapper class for the EarthFormer model (implemented via CuboidTransformerModel)
    for precipitation nowcasting.

    This class handles the initialization, checkpoint loading, device management,
    and inference for the EarthFormer model, making it compatible with a broader
    evaluation or prediction pipeline. It also includes specific data
    preprocessing/postprocessing steps relevant to radar data.
    """
    def __init__(
        self, config: dict
    ):
        """
        Initializes the EarthFormerModel by instantiating the CuboidTransformerModel
        and loading a pre-trained checkpoint if specified.

        Parameters:
            config : dict
                A dictionary containing configuration parameters for the model.
                Expected keys include:
                - `model.checkpoint_path` (str, optional): Path to the PyTorch
                  checkpoint file (`.pt` or `.pth`) to load model weights from.
                - Other parameters specific to `CuboidTransformerModel`'s constructor
                  (passed via `**config.model`).
        """
        checkpoint_path = config.model.get('checkpoint_path', None)

        self.model = CuboidTransformerModel(**config.model)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
            model_state_dict = {
                key[len('torch_nn_module.'):] : val
                for key, val in checkpoint["state_dict"].items()
                if key.startswith("torch_nn_module.") and key[len('torch_nn_module.'):] in self.model.state_dict()
            }
            self.model.load_state_dict(model_state_dict, strict=False)

        self.model.eval()
        self.model.to(self.device)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs a forward pass (inference) with the EarthFormer model.

        This method handles necessary tensor shape permutations, device transfer,
        inference in `no_grad` context, and post-processing of the output.

        Parameters:
            x : torch.Tensor
                The input radar data tensor. Expected shape: (Batch, Channels, Time, Height, Width).
                This will be permuted to (Batch, Time, Height, Width, Channels) for the model.
            y : torch.Tensor
                The target radar data tensor (passed through, but also post-processed).
                Expected shape: (Batch, Channels, Time, Height, Width).

        Returns:
            tuple[np.ndarray, np.ndarray]
                A tuple containing:
                - y_np: The processed target data as a NumPy array.
                - y_hat_np: The model's prediction as a NumPy array.
                Both outputs are in the original data scale (after `data_prep` undo).
        """
        # Shape (B, C, T, W, H) -> Earthformer Shape (B, T, W, H, C)
        x = x.permute(0, 2, 3, 4, 1)

        x = x.to(self.device)

        with torch.no_grad():
            y_hat = self.model(x)

        # Earthformer Shape (B, T, W, H, C) -> (B, C, T, W, H)
        y_hat = y_hat.permute(0, 4, 1, 2, 3)

        y_hat = y_hat.detach().cpu().numpy()

        # Convert dBz to mm/h for evaluation.
        data_prep(y_hat, convert_to_dbz=True, undo=True)
        data_prep(y, convert_to_dbz=True, undo=True)

        return y, y_hat