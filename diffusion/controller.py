"""Classes and functions for controlling the diffusion training process."""
from typing import (
    Any,
    Dict,
    Optional,
)

import torch
from torch import Tensor, nn

from diffusion.diffusor import DDIMDiffusor
from diffusion.model import Unet
from diffusion.schedule import ScheduleFactory


class DiffusionController:
    """Class for handling the diffusion process."""

    def __init__(
        self,
        model_params: Dict,
        schedule_params: Dict,
        diffusor_params: Dict,
        device: torch.device,
        loss_func: str = 'l1',
        ddp: bool = False,
    ) -> None:
        """Initialize DiffusionController."""
        self.device = device
        self.model = Unet(**model_params).to(device)
        schedule_params['device'] = self.device
        self.schedule = ScheduleFactory.get_schedule(**schedule_params)
        self.diffusor = DDIMDiffusor(
            model=self.model,
            schedule=self.schedule,
            device=self.device,
            **diffusor_params
        )

        self.loss_func = DiffusionController._get_loss_func(loss_func)
        self.ddp_used = ddp

    def add_noise(self, x_start: Tensor, t: Tensor, do_scaling: bool = True) -> Tensor:
        """Corrupt one or more images with noise according to a timestep."""
        orig_device = x_start.device
        x_start = x_start.to(self.device)
        t = t.to(self.device)

        if do_scaling:
            x_start = x_start * 2 - 1

        x_noised = self.diffusor.q_sample(x_start, t)
        x_noised.clamp_(-1, 1)

        if do_scaling:
            x_noised = (x_noised + 1) / 2

        return x_noised.to(orig_device)

    @torch.no_grad()
    def generate_noise(self, x_like: Tensor) -> Tensor:
        """Generate noise from the noise function."""
        return self.diffusor.noise_fn(x_like)

    def generate(
        self,
        img_res: int,
        batch_size: int,
        log_every: Optional[int] = None,
        do_scaling: bool = True,
    ) -> Tensor:
        """Generate a batch of new images."""
        self.diffusor.model.eval()
        shape = (batch_size, self.get_model_obj().channels, img_res, img_res)

        if log_every is None:
            res = self.diffusor.p_sample_loop(shape)
        else:
            res = self.diffusor.p_sample_loop_with_steps(shape, log_every)

        if do_scaling:
            res.clamp_(-1, 1)
            res = (res + 1) / 2
        self.diffusor.model.train()
        return res

    def get_model_obj(self) -> nn.Module:
        """Return model object. Helps in accessing attributes in a DDP setting."""
        if self.ddp_used:
            return self.model.module
        else:
            return self.model

    def get_model_params(self) -> Any:
        """Return model parameters."""
        return self.model.parameters()

    @staticmethod
    def _get_loss_func(fn_name: str) -> nn.Module:
        """Get loss function module corresponding to string identifier."""
        if fn_name == 'l1':
            return nn.L1Loss()
        elif fn_name == 'l2':
            return nn.MSELoss()
        elif fn_name == 'huber':
            return nn.SmoothL1Loss()
        else:
            raise ValueError('Loss function name "{}" is unknown.'.format(fn_name))

    def get_loss(
        self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        """Execute training procedure for a batch of images."""
        if noise is None:
            noise = self.generate_noise(x_start)

        x_noisy = self.diffusor.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)

        return self.loss_func(noise, predicted_noise)
