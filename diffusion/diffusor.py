"""Classes and functions for diffusion processes."""
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from tqdm import tqdm

from diffusion.schedule import BaseSchedule


class Diffusor:
    """Class for modelling the diffusion process."""

    def __init__(
        self,
        model: nn.Module,
        schedule: BaseSchedule,
        device: Optional[torch.device] = None,
        clip_denoised: bool = True,
    ) -> None:
        """Initialize Diffusor."""
        self.model = model
        self.schedule = schedule
        self.clip_denoised = clip_denoised

        if device is None:
            self.device = schedule.device
        else:
            self.device = device

    @staticmethod
    def extract_vals(a: Tensor, t: Tensor, x_shape: Tuple) -> Tensor:
        """Extract timestep values from tensor and reshape to target dims."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        """
        Sample from forward process q.

        Given an initial `x_start` and a timestep `t, return perturbed images `x_t`.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = Diffusor.extract_vals(
            self.schedule.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = Diffusor.extract_vals(
            self.schedule.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        """Subtract noise from x_t over variance schedule."""
        sqrt_recip_alphas_cumprod_t = Diffusor.extract_vals(
            self.schedule.sqrt_recip_alphas_cumprod, t, x_t.shape
        )
        sqrt_recipm1_alphas_cumprod_t = Diffusor.extract_vals(
            self.schedule.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

        x_rec = (
            sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
        )

        return x_rec

    def q_posterior(
        self, x_start: Tensor, x_t: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute posterior q."""
        posterior_mean_coef1 = Diffusor.extract_vals(
            self.schedule.post_mean_coef1, t, x_t.shape
        )
        posterior_mean_coef2 = Diffusor.extract_vals(
            self.schedule.post_mean_coef2, t, x_t.shape
        )
        post_var = Diffusor.extract_vals(self.schedule.post_var, t, x_t.shape)
        posterior_log_var_clipped = Diffusor.extract_vals(
            self.schedule.post_log_var_clipped, t, x_t.shape
        )

        posterior_mean = posterior_mean_coef1 * x_start + posterior_mean_coef2 * x_t
        return posterior_mean, post_var, posterior_log_var_clipped

    def p_mean_variance(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute mean and variance for reverse process."""
        noise_pred = self.model(x, t)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        return self.q_posterior(x_start=x_recon, x_t=x, t=t)

    @torch.no_grad()
    def p_sample(self, x: Tensor, t: Tensor) -> Tensor:
        """Sample from reverse process."""
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t)
        noise = torch.randn_like(x)

        if t[0] == 0:
            return model_mean
        else:
            return model_mean + (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple) -> Tensor:
        """Initiate generation process."""
        batch_size = shape[0]
        img = torch.randn(shape, device=self.device)

        pbar = tqdm(
            iterable=reversed(range(0, self.schedule.timesteps)),
            desc='sampling loop time step',
            total=self.schedule.timesteps,
            leave=False,
        )
        for i in pbar:
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t)
        return img

    @torch.no_grad()
    def p_sample_loop_with_steps(self, shape: Tuple, log_every_t: int) -> Tensor:
        """Initiate generation process and return intermediate steps."""
        batch_size = shape[0]
        img = torch.randn(shape, device=self.device)

        result = [img]

        pbar = tqdm(
            iterable=reversed(range(0, self.schedule.timesteps)),
            desc='sampling loop time step',
            total=self.schedule.timesteps,
            leave=False,
        )
        for i in pbar:
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, t)

            if i % log_every_t == 0 or i == self.schedule.timesteps - 1:
                result.append(img)

        return torch.stack(result)


class SR3Diffusor(Diffusor):
    """Class for modelling the diffusion process in SR3."""

    def p_mean_variance(  # type: ignore
        self, x: Tensor, sr: Tensor, t: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute mean and variance for reverse process."""
        x_in = torch.cat([x, sr], dim=1)
        noise_pred = self.model(x_in, t)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        return self.q_posterior(x_start=x_recon, x_t=x, t=t)

    @torch.no_grad()
    def p_sample(self, x: Tensor, sr: Tensor, t: Tensor) -> Tensor:
        """Sample from reverse process."""
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, sr=sr, t=t)
        noise = torch.randn_like(x)

        if t[0] == 0:
            return model_mean
        else:
            return model_mean + (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, sr: Tensor) -> Tensor:
        """Initiate generation process."""
        batch_size = sr.shape[0]
        img = torch.randn(sr.shape, device=self.device)

        pbar = tqdm(
            iterable=reversed(range(0, self.schedule.timesteps)),
            desc='sampling loop time step',
            total=self.schedule.timesteps,
            leave=False,
        )
        for i in pbar:
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, sr, t)
        return img

    @torch.no_grad()
    def p_sample_loop_with_steps(self, sr: Tensor, log_every_t: int) -> Tensor:
        """Initiate generation process and return intermediate steps."""
        batch_size = sr.shape[0]
        img = torch.randn(sr.shape, device=self.device)

        result = [img]

        pbar = tqdm(
            iterable=reversed(range(0, self.schedule.timesteps)),
            desc='sampling loop time step',
            total=self.schedule.timesteps,
            leave=False,
        )
        for i in pbar:
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(img, sr, t)

            if i % log_every_t == 0 or i == self.schedule.timesteps - 1:
                result.append(img)

        return torch.stack(result)
