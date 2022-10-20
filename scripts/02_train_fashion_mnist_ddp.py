"""Train a diffusion model on FashionMNIST."""
import os
from typing import (
    Dict,
    Final,
    List,
)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, DistributedSampler
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
    'epochs': 100,
    'batch_size': 64,
    'num_workers': 4,
    'learning_rate': 1e-3,
    'sample_freq': 10,
    'sample_size': 8,
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
        'name': 'cosine',
        'timesteps': 1000,
    },
    'diffusor_params': {},
    'loss_func': 'l1',
}


def run(rank: int, world_size: int, cfg: Dict) -> None:
    """Kick off training."""
    setup(rank, world_size)
    device = torch.device(rank)

    # ----------------------------------------------------------------------------------
    # CREATE DIFFUSION MODEL
    # ----------------------------------------------------------------------------------
    diff = DiffusionController(
        model_params=cfg['model_params'],
        schedule_params=cfg['schedule_params'],
        diffusor_params=cfg['diffusor_params'],
        device=device,
        loss_func=cfg['loss_func'],
        ddp=True,
    )

    diff.model = DDP(diff.model, device_ids=[rank])

    optimizer = Adam(diff.get_model_params(), lr=cfg['learning_rate'])

    # ----------------------------------------------------------------------------------
    # DATASETS & DATALOADERS
    # ----------------------------------------------------------------------------------

    transform = Compose(
        [Grayscale(), RandomHorizontalFlip(), ToTensor(), Lambda(rescale)]
    )

    train_dataset = FashionMNIST(
        root='.data',
        train=True,
        transform=transform,
        download=True,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        sampler=train_sampler,
        persistent_workers=True,
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
        rank=rank,
        world_size=world_size,
        cfg=cfg,
    )


def rescale(x: Tensor) -> Tensor:
    """Rescale image tensor from [0, 1] to [-1, 1]."""
    return (x * 2) - 1


def train(
    diff: DiffusionController,
    optimizer: Optimizer,
    train_loader: DataLoader,
    epochs: int,
    rank: int,
    world_size: int,
    cfg: Dict,
) -> Dict[str, List[float]]:
    """Execute training procedure."""
    device = diff.device
    scaler = GradScaler()

    # ----------------------------------------------------------------------------------
    # METRICS
    # ----------------------------------------------------------------------------------
    metric_train_loss = MeanMetric(compute_on_step=False).to(device)

    # We also define a dictionary that keeps track over all computed metrics.
    metrics: Dict[str, List] = {
        'train/loss': [],
    }

    # ----------------------------------------------------------------------------------
    # START TRAINING PROCEDURE
    # ----------------------------------------------------------------------------------
    steps = 1
    for ep in range(1, epochs + 1):

        # ------------------------------------------------------------------------------
        # TRAINING LOOP
        # ------------------------------------------------------------------------------
        for x, _ in tqdm(train_loader, leave=False):
            x = x.to(device)
            t = torch.randint(
                0, diff.schedule.timesteps, (x.shape[0],), device=device
            ).long()

            with autocast():
                batch_loss = diff.get_loss(x, t)

            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            steps += 1

            metric_train_loss(batch_loss)

        # ------------------------------------------------------------------------------
        # METRIC COMPUTATION
        # ------------------------------------------------------------------------------
        ep_train_loss = float(metric_train_loss.compute())
        metric_train_loss.reset()

        # Add current metrics to history.
        metrics['train/loss'].append(ep_train_loss)

        # ------------------------------------------------------------------------------
        # TRAINING PROGRESS & MODEL SAVING
        # ------------------------------------------------------------------------------
        if rank == 0:
            # Print training progress.
            print('EP: {:3} | LOSS: T {:.3f} '.format(ep, ep_train_loss))

            # Save model
            log_dir = cfg['log_dir']
            os.makedirs(log_dir, exist_ok=True)
            torch.save(
                diff.model.module.state_dict(), os.path.join(log_dir, 'model.pt')
            )

        # ------------------------------------------------------------------------------
        # EVALUATION SAMPLES
        # ------------------------------------------------------------------------------

        if ep % cfg['sample_freq'] == 0:
            if rank == 0:
                print('\n--> GENERATING SAMPLES FROM MODEL <--')
            imgs = diff.generate(
                img_res=28,
                batch_size=cfg['sample_size'],
                log_every=diff.schedule.timesteps // 10,
            ).detach()

            tensor_list = [
                torch.zeros(imgs.shape, dtype=torch.float, device=device)
                for _ in range(world_size)
            ]
            dist.all_gather(tensor_list=tensor_list, tensor=imgs)

            if rank == 0:
                gathered_imgs = torch.cat(tensor_list, dim=1)
                grid_path = os.path.join(
                    cfg['log_dir'], 'sample_ep_{}'.format(str(ep).zfill(3))
                )
                save_sampling_grid(gathered_imgs, grid_path)

            dist.barrier()

    return metrics


def save_sampling_grid(imgs: Tensor, save_path: str) -> None:
    """Save a grid of samples to a file."""
    imgs = imgs.detach().cpu()
    n_cols = imgs.shape[0]
    imgs = imgs.permute(1, 0, 2, 3, 4).reshape(-1, 1, 28, 28)
    plot_image(imgs, ncols=n_cols, show=False, save_path=save_path)


def setup(rank: int, world_size: int) -> None:
    """Initialize environment for DDP training."""
    # Set adress and port for node communication.
    # We simply choose localhost here, which should suffice in most cases.
    os.environ['MASTER_ADDR'] = 'localhost'

    # The master port needs to be free, which could be an issue in a multi-user setting.
    # as we're using slurm to deploy our jobs, we can use the last digits of the slurm
    # job id to get our port. Otherwise, we use a default.
    try:
        default_port = os.environ['SLURM_JOB_ID']
        default_port = default_port[-4:]

        # All ports should be in the 10k+ range
        default_port = int(default_port) + 15000  # type: ignore

    except Exception:
        default_port = 12910  # type: ignore

    os.environ['MASTER_PORT'] = str(default_port)

    # Initialize distributed process group.
    # torch offers a few backends, but usually NCCL is the best choice.
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    print('DISTRIBUTED WORKER {} of {} INITIALIZED.'.format(rank + 1, world_size))


def cleanup():
    """Clean up process groups from DDP training."""
    dist.destroy_process_group()


def main() -> None:
    """Execute main func."""
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, BASE_CONFIG), nprocs=world_size)


if __name__ == '__main__':
    main()
