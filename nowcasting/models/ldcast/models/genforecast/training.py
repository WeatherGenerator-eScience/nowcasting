import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from ..diffusion import diffusion


def setup_genforecast_training(
    model,
    autoencoder,
    context_encoder,
    model_dir,
    lr=1e-5
):
    # For mixed presision
    torch.set_float32_matmul_precision('medium')

    ldm = diffusion.LatentDiffusion(model, autoencoder, 
        context_encoder=context_encoder, lr=lr)

    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if (num_gpus > 0) else "cpu"
    devices = torch.cuda.device_count() if (accelerator == "gpu") else 1

    early_stopping = pl.callbacks.EarlyStopping(
        "val/loss_ema", patience=20, verbose=True, check_finite=False
    )
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{val_loss_ema:.4f}",
        monitor="val/loss_ema",
        every_n_epochs=1,
        save_top_k=3
    )
    callbacks = [early_stopping]

    logger = TensorBoardLogger("/vol/csedu-nobackup/project/mrobben/nowcasting/results/tb_logs", name="LDCast_experiment")

    trainer = pl.Trainer(
        accelerator='auto',
        devices=devices,
        max_epochs=1000,
        strategy='auto',
        logger=logger,
        callbacks=callbacks,
        precision='16-mixed',
        limit_train_batches=500,
        limit_val_batches=50,
        enable_checkpointing=True,
        # accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    return (ldm, trainer)
