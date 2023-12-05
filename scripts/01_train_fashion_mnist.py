"""Train a diffusion model on FashionMNIST."""
import os
from typing import (
    Dict,
    Final,
    List,
)

import torch
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torchvision.datasets import FashionMNIST
from torchvision.transforms import (
    Compose,
    Grayscale,
    Lambda,
    RandomHorizontalFlip,
    ToTensor,
)
from tqdm import tqdm

from diffusion.controller import DiffusionController
from diffusion.utils import plot_image

BASE_CONFIG: Final = {
    'epochs': 1000,
    'batch_size': 128,
    'num_workers': 8,
    'learning_rate': 1e-4,
    'sample_freq': 5,
    'sample_size': 4,
    'log_dir': 'logs/fmnist',
    'model_params': {
        'dim': 32,
        'channels': 1,
        'dim_mults': (
            1,
            2,
            4,
        ),
    },
    'schedule_params': {
        'name': 'linear',
        'timesteps': 500,
        'beta_start': 0.0001,
        'beta_end': 0.04,
    },
    'diffusor_params': {'noise_fn': 'simplex'},
    'loss_func': 'l1',
}


def run(cfg: Dict) -> None:
    """Kick off training."""
    device = torch.device('cuda')

    # ----------------------------------------------------------------------------------
    # CREATE DIFFUSION MODEL
    # ----------------------------------------------------------------------------------
    diff = DiffusionController(
        model_params=cfg['model_params'],
        schedule_params=cfg['schedule_params'],
        diffusor_params=cfg['diffusor_params'],
        device=device,
        loss_func=cfg['loss_func'],
    )

    optimizer = Adam(diff.get_model_params(), lr=cfg['learning_rate'])

    # ----------------------------------------------------------------------------------
    # DATASETS & DATALOADERS
    # ----------------------------------------------------------------------------------

    transform = Compose(
        [Grayscale(), RandomHorizontalFlip(), ToTensor(), Lambda(lambda t: (t * 2) - 1)]
    )

    train_dataset = FashionMNIST(
        root='.data',
        train=True,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
    )

    # ----------------------------------------------------------------------------------
    # CONDUCT TRAINING
    # ----------------------------------------------------------------------------------
    train(
        diff=diff,
        optimizer=optimizer,
        train_loader=train_loader,
        epochs=cfg['epochs'],
        cfg=cfg,
    )


def train(
    diff: DiffusionController,
    optimizer: Optimizer,
    train_loader: DataLoader,
    epochs: int,
    cfg: Dict,
) -> Dict[str, List[float]]:
    """Execute training procedure."""
    device = diff.device

    # ----------------------------------------------------------------------------------
    # METRICS
    # ----------------------------------------------------------------------------------
    metric_train_loss = MeanMetric().to(device)

    # We also define a dictionary that keeps track over all computed metrics.
    metrics: Dict[str, List] = {
        'train/loss': [],
    }

    # ----------------------------------------------------------------------------------
    # START TRAINING PROCEDURE
    # ----------------------------------------------------------------------------------
    for ep in range(1, epochs + 1):
        # ------------------------------------------------------------------------------
        # TRAINING LOOP
        # ------------------------------------------------------------------------------
        for x, _ in tqdm(train_loader, leave=False):
            x = x.to(device)
            t = torch.randint(
                0, diff.schedule.timesteps, (x.shape[0],), device=device
            ).long()

            batch_loss = diff.get_loss(x, t)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            metric_train_loss(batch_loss)

        # ------------------------------------------------------------------------------
        # METRIC COMPUTATION
        # ------------------------------------------------------------------------------
        ep_train_loss = float(metric_train_loss.compute())
        metric_train_loss.reset()

        # Add current metrics to history.
        metrics['train/loss'].append(ep_train_loss)

        # Print training progress.
        print('EP: {:3} | LOSS: T {:.3f} '.format(ep, ep_train_loss))

        # Save model
        log_dir = cfg['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        torch.save(diff.model.state_dict(), os.path.join(log_dir, 'model.pt'))

        # Sample some images
        if ep % cfg['sample_freq'] == 0:
            print('--> GENERATING SAMPLES FROM MODEL <--')
            imgs = diff.generate(
                img_res=28, batch_size=cfg['sample_size'], log_every=20
            )
            grid_path = os.path.join(log_dir, 'sample_ep_{}'.format(str(ep).zfill(3)))
            save_sampling_grid(imgs, grid_path)

    return metrics


def save_sampling_grid(imgs: Tensor, save_path: str) -> None:
    """Save a grid of samples to a file."""
    imgs = imgs.detach().cpu()
    n_cols = imgs.shape[0]
    imgs = imgs.permute(1, 0, 2, 3, 4).reshape(-1, 1, 28, 28)
    plot_image(imgs, ncols=n_cols, show=False, save_path=save_path)


def main() -> None:
    """Execute main func."""
    run(BASE_CONFIG)


if __name__ == '__main__':
    main()
