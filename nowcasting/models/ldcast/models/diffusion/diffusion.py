"""
From https://github.com/CompVis/latent-diffusion/main/ldm/models/diffusion/ddpm.py
Pared down to simplify code.

The original file acknowledges:
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
"""

import torch
import torch.nn as nn
import numpy as np
import torchvision.utils
import pytorch_lightning as pl
from contextlib import contextmanager
from functools import partial
 
from .utils import make_beta_schedule, extract_into_tensor, noise_like, timestep_embedding
from .ema import LitEma
from .plms import PLMSSampler
from ..blocks.afno import PatchEmbed3d, PatchExpand3d, AFNOBlock3d


class LatentDiffusion(pl.LightningModule):
    def __init__(self,
        model,
        autoencoder,
        context_encoder=None,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        use_ema=True,
        lr=1e-4,
        lr_warmup=0,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        parameterization="eps",  # all assuming fixed variance schedules
        val_num_diffusion_iters=50,
    ):
        super().__init__()
        self.model = model
        self.autoencoder = autoencoder.requires_grad_(False)
        self.conditional = (context_encoder is not None)
        self.context_encoder = context_encoder
        self.lr = lr
        self.lr_warmup = lr_warmup
        self.val_num_diffusion_iters = val_num_diffusion_iters

        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)

        self.register_schedule(
            beta_schedule=beta_schedule, timesteps=timesteps,
            linear_start=linear_start, linear_end=linear_end, 
            cosine_s=cosine_s
        )

        self.loss_type = loss_type

        self.sample_images = None
        self.sampler = PLMSSampler(self)

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):

        betas = make_beta_schedule(
            beta_schedule, timesteps,
            linear_start=linear_start, linear_end=linear_end,
            cosine_s=cosine_s
        )
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def apply_model(self, x_noisy, t, cond=None, return_ids=False):
        if self.conditional:
            cond = self.context_encoder(cond)
        with self.ema_scope():
            return self.model(x_noisy, t, context=cond)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None, context=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t, context=context)

        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")

        return self.get_loss(model_out, target, mean=False).mean()

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def shared_step(self, batch):
        (x,y) = batch
        y = self.autoencoder.encode(y)[0]
        context = self.context_encoder(x) if self.conditional else None
        return self(y, context=context)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train/lr", lr, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        with self.ema_scope():
            loss_ema = self.shared_step(batch)

        if batch_idx == 0 and self.global_rank == 0:
            (x, y) = batch
            x_s = x[0][0][0:1]
            x_t = x[0][1][0:1]
            y = y[0:1]
            x = [[x_s, x_t]]

            batch, channel, time, width, height = y.shape
            gen_shape = (self.autoencoder.hidden_width, time//4, width//4, height//4)

            with self.ema_scope():
                 # The sampler needs the model (self), steps, batch_size, shape, condition
                 # Condition should be the encoded context for the *first* batch item
                latent_sample, _ = self.sampler.sample(
                    S=self.val_num_diffusion_iters,
                    batch_size=1, # Generate only one sample
                    shape=gen_shape, # Shape without batch dim for sampler
                    conditioning=x, # Pass encoded context for the sample
                    verbose=False, # Disable inner progress bar
                )

            pred_img = self.autoencoder.decode(latent_sample)

            input_img = x_s[0].permute(1,0,2,3).cpu() # Input frames for the sample
            target_img = y[0].permute(1,0,2,3).cpu() # Target frames for the sample
            pred_img = pred_img[0].permute(1,0,2,3).cpu() # Predicted frames (remove batch dim)

            self.sample_images = (input_img, target_img, pred_img)
            print(f"Validation Sampling: Generated sample. Input: {input_img.shape}, Target: {target_img.shape}, Prediction: {pred_img.shape}")


        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log("val/loss", loss, **log_params)
        self.log("val/loss_ema", loss, **log_params)

    def on_validation_epoch_end(self):

         if self.sample_images is not None and self.global_rank == 0:
            input_img, target_img, pred_img = self.sample_images

            print(f"on_validation_epoch_end: Found sample data on rank {self.global_rank}. Creating grid...")
            print(f"Input shape: {input_img.shape}, Target shape: {target_img.shape}, Prediction shape: {pred_img.shape}")

            # Get dimensions AFTER ensuring channel dim exists
            T_in, C_in, H_in, W_in = input_img.shape
            T_out, C_out, H_out, W_out = target_img.shape

            # Create visualization grid
            last_input = input_img[-1].unsqueeze(0)  # Last input frame [1, C, H, W]

            # Interleave target and prediction frames
            interleaved = []
            for t in range(T_out):
                 # Add target and prediction side-by-side for each timestep
                 # Ensure they have the same C, H, W
                 # Clamp values to [0, 1] for visualization
                target_frame = target_img[t].clamp(0, 1)
                pred_frame = pred_img[t].clamp(0, 1)
                interleaved.append(target_frame)
                interleaved.append(pred_frame)


            # Combine: Last Input | Target_0 | Pred_0 | Target_1 | Pred_1 | ...
            # Stack interleaved first, then concatenate with last input
            if interleaved: # Only proceed if there are output frames
                interleaved_stack = torch.stack(interleaved) # [2*T_out, C, H, W]
                all_frames = torch.cat([last_input, interleaved_stack], dim=0) # [1 + 2*T_out, C, H, W]

                # Create grid (adjust nrow based on how you want to display)
                grid = torchvision.utils.make_grid(all_frames, nrow=1 + 2*T_out, normalize=False) # Already clamped

                # Log to TensorBoard
                self.logger.experiment.add_image(
                    "forecast_samples",
                    grid,
                    global_step=self.current_epoch,
                )
                print(f"on_validation_epoch_end: Logged image grid to TensorBoard for epoch {self.current_epoch}.")
            else:
                 print(f"on_validation_epoch_end: No output frames (T_out={T_out}) found in sample data.")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
            betas=(0.5, 0.9), weight_decay=1e-3)
        # reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, patience=20, factor=0.1, verbose=True
        # )

        reduce_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-9  # Minimum learning rate
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val/loss_ema",
                "frequency": 1,
            },
        }

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def optimizer_step(
        self, 
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
        **kwargs    
    ):
        if self.trainer.global_step < self.lr_warmup:
            lr_scale = (self.trainer.global_step+1) / self.lr_warmup
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        super().optimizer_step(
            epoch, batch_idx, optimizer,
            optimizer_closure=optimizer_closure,
            **kwargs
        )
    
