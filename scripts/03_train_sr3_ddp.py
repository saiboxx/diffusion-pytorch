"""Train a diffusion model for super resolution."""
import os
from typing import (
    Dict,
    Final,
    List,
)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import MeanMetric
from torchvision.transforms import (
    Compose,
    Grayscale,
    Lambda,
    ToTensor,
)
from tqdm import tqdm

from diffusion.controller import DiffusionController
from diffusion.data import MaCheXDataset
from diffusion.diffusor import SR3Diffusor
from diffusion.utils import plot_image

BASE_CONFIG: Final = {
    'epochs': 250,
    'batch_size': 8,
    'num_workers': 16,
    'learning_rate': 5e-5,
    'print_freq': 50,
    'save_freq': 1000,
    'sample_freq': 5000,
    'sample_size': 4,
    'resume_checkpoint': 'logs/sr3/model.pt',
    'data_root': '/data/core-rad/machex',
    'log_dir': 'logs/sr3',
    'model_params': {
        'dim': 16,
        'channels': 2,
        'out_dim': 1,
        'dim_mults': (1, 2, 4, 8, 16, 32, 32, 32),
    },
    'schedule_params': {
        'name': 'cosine',
        'timesteps': 2000,
    },
    'diffusor_params': {},
    'loss_func': 'l1',
}


def run(rank: int, world_size: int, cfg: Dict) -> None:
    """Kick off training."""
    setup(rank, world_size)
    device = torch.device(rank)

    if rank == 0:
        wandb.init(project='chex-sr3', config=cfg)

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

    if cfg['resume_checkpoint'] is not None and os.path.exists(
        cfg['resume_checkpoint']
    ):
        state_dict = torch.load(cfg['resume_checkpoint'], map_location=device)
        diff.model.module.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        cfg['resume_ep'] = state_dict['epoch']
        cfg['resume_steps'] = state_dict['steps']
        if rank == 0:
            print(
                '--> RETURNING FROM CHECKPOINT: EPOCH {} | STEP {}'.format(
                    cfg['resume_ep'], cfg['resume_steps']
                )
            )

    # ----------------------------------------------------------------------------------
    # DATASETS & DATALOADERS
    # ----------------------------------------------------------------------------------

    transforms = Compose([Grayscale(), ToTensor(), Lambda(rescale)])

    train_dataset = MaCheXDataset(
        root=cfg['data_root'],
        transforms=transforms,
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
) -> None:
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

    log_dir = cfg['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    # ----------------------------------------------------------------------------------
    # START TRAINING PROCEDURE
    # ----------------------------------------------------------------------------------
    steps = 0 if cfg.get('resume_steps') is None else cfg.get('resume_steps')
    ep_start = 1 if cfg.get('resume_ep') is None else cfg.get('resume_ep')
    for ep in range(ep_start, epochs + 1):  # type: ignore

        # ------------------------------------------------------------------------------
        # TRAINING LOOP
        # ------------------------------------------------------------------------------
        for batch in tqdm(train_loader, leave=False):
            # True high resolution ground truth
            x_hr = batch['HR'].to(device)
            # Interpolated low resolution as conditioning
            x_sr = batch['SR'].to(device)

            # Sample random timestep
            t = torch.randint(
                0, diff.schedule.timesteps, (x_hr.shape[0],), device=device
            ).long()

            # Create noisy HR image
            noise = torch.randn_like(x_hr)
            x_hr_noisy = diff.diffusor.q_sample(x_start=x_hr, t=t, noise=noise)

            # Concatenate noisy HR image with SR image
            x_in = torch.cat([x_hr_noisy, x_sr], dim=1)

            # Predict noise on HR image
            with autocast():
                # Predict noise on HR image
                predicted_noise = diff.model(x_in, t)
                # Compute loss
                batch_loss = diff.loss_func(noise, predicted_noise)

            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            metric_train_loss(batch_loss)
            steps += 1  # type: ignore

            # --------------------------------------------------------------------------
            # PRINT INTERMEDIATE PROGRESS
            # --------------------------------------------------------------------------

            if steps % cfg['print_freq'] == 0 and rank == 0:
                tqdm.write(
                    'STEP {:7} | BATCH LOSS: {:.3f}'.format(steps, float(batch_loss))
                )
                wandb.log(
                    {'batch_loss': float(batch_loss), 'epoch': ep, 'step': steps},
                    step=steps,
                )

            # --------------------------------------------------------------------------
            # SAVE MODEL
            # --------------------------------------------------------------------------
            if steps % cfg['save_freq'] == 0 and rank == 0:
                torch.save(
                    {
                        'model': diff.model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': ep,
                        'steps': steps,
                    },
                    os.path.join(log_dir, 'model.pt'),
                )
                tqdm.write('---> CHECKPOINT SAVED <--- ')

            # --------------------------------------------------------------------------
            # EVALUATION SAMPLES
            # --------------------------------------------------------------------------

            if steps % cfg['sample_freq'] == 0:
                if rank == 0:
                    tqdm.write('---> GENERATING SAMPLES FROM MODEL <---')
                with autocast():
                    diff.model.eval()
                    diffusor = SR3Diffusor(
                        model=diff.model, schedule=diff.schedule, device=device
                    )
                    x_eval = x_sr[: min(cfg['sample_size'], len(x_sr))]
                    imgs = diffusor.p_sample_loop_with_steps(
                        sr=x_eval, log_every_t=diff.schedule.timesteps // 4
                    )
                    diff.model.train()

                imgs = imgs.detach().float()
                imgs.clamp_(-1, 1)
                imgs = (imgs + 1) / 2

                tensor_list = [
                    torch.zeros(imgs.shape, dtype=torch.float, device=device)
                    for _ in range(world_size)
                ]
                dist.all_gather(tensor_list=tensor_list, tensor=imgs)

                if rank == 0:
                    gathered_imgs = torch.cat(tensor_list, dim=1)
                    grid_path = os.path.join(
                        cfg['log_dir'], 'sample_step_{}'.format(str(steps).zfill(6))
                    )
                    save_sampling_grid(gathered_imgs, grid_path)

                dist.barrier()

        # ------------------------------------------------------------------------------
        # METRIC COMPUTATION
        # ------------------------------------------------------------------------------
        ep_train_loss = float(metric_train_loss.compute())
        metric_train_loss.reset()

        # Add current metrics to history.
        metrics['train/loss'].append(ep_train_loss)

        if rank == 0:
            print('EP: {:3} | LOSS: T {:.3f} '.format(ep, ep_train_loss))

            # Save model and exit.
            # This is due to time constraints in a SLURM Cluster.
            # A follow-up job is triggered externally to continue training.
            torch.save(
                {
                    'model': diff.model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': ep + 1,
                    'steps': steps,
                },
                os.path.join(log_dir, 'model.pt'),
            )

        cleanup()
        return


def save_sampling_grid(imgs: Tensor, save_path: str) -> None:
    """Save a grid of samples to a file."""
    imgs = imgs.detach().cpu()
    n_cols = imgs.shape[0]
    imgs = imgs.permute(1, 0, 2, 3, 4).reshape(-1, 1, 1024, 1024)
    plot_image(imgs, fig_size=(25, 25), ncols=n_cols, show=False, save_path=save_path)


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
    wandb.finish()
    dist.destroy_process_group()


def main() -> None:
    """Execute main func."""
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, BASE_CONFIG), nprocs=world_size)


if __name__ == '__main__':
    main()
