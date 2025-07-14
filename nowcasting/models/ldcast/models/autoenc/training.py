import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from . import autoenc


def setup_autoenc_training(
    encoder,
    decoder,
    model_dir,
    encoded_channels=64,
    hidden_width=32,
):
    autoencoder = autoenc.AutoencoderKL(encoder, decoder, encoded_channels=encoded_channels, hidden_width=hidden_width)

    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if (num_gpus > 0) else "cpu"
    devices = torch.cuda.device_count() if (accelerator == "gpu") else 1

    early_stopping = pl.callbacks.EarlyStopping(
        "val/rec_loss", patience=6, verbose=True
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{val_rec_loss:.4f}",
        monitor="val/rec_loss",
        every_n_epochs=1,
        save_top_k=3
    )
    callbacks = [early_stopping, checkpoint]

    logger = TensorBoardLogger("/vol/csedu-nobackup/project/mrobben/nowcasting/results/", name="LDCast_experiment")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=1000,
        strategy='auto',
        logger=logger,
        limit_train_batches=100,
        limit_val_batches=30,
        # callbacks=callbacks
    )

    return (autoencoder, trainer)
