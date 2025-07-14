import torch
from torch import nn
import lightning as L

from ..distributions import kl_from_standard_normal, ensemble_nll_normal
from ..distributions import sample_from_standard_normal

from .encoder import SimpleConvDecoder, SimpleConvEncoder

class AutoencoderKL(L.LightningModule):
    def __init__(
        self, config
    ):
        super(AutoencoderKL, self).__init__()
        enc_params = config.model.get('enc_params', {})
        dec_params = config.model.get('enc_params', {})
        kl_weight = config.model.get('kl_weight', 0.01)
        encoded_channels = config.model.get('encoded_channels', 64)
        hidden_width = config.model.get('hidden_width', 32)
        self.config = config

        self.encoder = SimpleConvEncoder(**enc_params)
        self.decoder = SimpleConvDecoder(**dec_params)
        self.hidden_width = config.model.hidden_width
        self.to_moments = nn.Conv3d(encoded_channels, 2*hidden_width,
            kernel_size=1)
        self.to_decoder = nn.Conv3d(hidden_width, encoded_channels,
            kernel_size=1)
        self.kl_weight = kl_weight

    def encode(self, x):
        h = self.encoder(x)
        (mean, log_var) = torch.chunk(self.to_moments(h), 2, dim=1)
        return (mean, log_var)

    def decode(self, z):
        z = self.to_decoder(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        (mean, log_var) = self.encode(input)
        if sample_posterior:
            z = sample_from_standard_normal(mean, log_var)
        else:
            z = mean
        dec = self.decode(z)
        return (dec, mean, log_var)

    def _loss(self, batch):
        if isinstance(batch, tuple):
            (x, y) = batch
        else:
            x = batch
            y = x.clone()

        (y_pred, mean, log_var) = self.forward(x)

        rec_loss = (y-y_pred).abs().mean()
        kl_loss = kl_from_standard_normal(mean, log_var)

        total_loss = rec_loss + self.kl_weight * kl_loss

        return (total_loss, rec_loss, kl_loss)

    def training_step(self, batch, batch_idx):
        total_loss, rec_loss, kl_loss = self._loss(batch)
        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/rec_loss", rec_loss, prog_bar=True)
        self.log("train/kl_loss", kl_loss, prog_bar=True)

        return total_loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split="val"):
        (total_loss, rec_loss, kl_loss) = self._loss(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log(f"{split}/loss", total_loss, **log_params)
        self.log(f"{split}/rec_loss", rec_loss.mean(), **log_params)
        self.log(f"{split}/kl_loss", kl_loss, **log_params)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        config = self.config.optimizer

        optimizer = torch.optim.AdamW(self.parameters(), lr=config.lr,
            betas=config.betas, weight_decay=config.weight_decay)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config.patience, factor=config.factor
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": config.monitor,
                "frequency": 1,
            },
        }