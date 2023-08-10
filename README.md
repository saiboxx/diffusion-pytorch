# Denoising Diffusion Models in Pytorch - Another Implementation

This repository contains a custom implementation of DDPM and SR3 in plain PyTorch.
The diffusion process is inspired by the code of [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion).
The U-net is based on [lucidrains DDPM implementation](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main).

The logging will be done with Weights & Biases.

Dependencies can be installed by pip, e.g. like:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example scripts

There are a few scripts available in the `scripts` directory.
Configuration dictionaries are contained within the file.
The resolution of the synthesized images will be the same as the one from the input dataset.

The classic DDPM can be trained over `01_train_fashion_mnist.py`.
As the name suggests, the base dataset for this is FashionMNIST, but this can easily be exchanged by every
standard (torchvision) dataset.
`02_train_fashion_mnist_ddp.py` does exactly the same but uses DDP for multi-GPU training.

This repository is one basis for the paper [Cascaded Latent Diffusion Models for High-Resolution Chest X-ray Synthesis](https://link.springer.com/chapter/10.1007/978-3-031-33380-4_14).
A DDPM that utilizes the [MaCheX dataset](https://github.com/saiboxx/machex) can be trained in via `04_train_chex_ddpm_ddp.py`.

Further, the super-resolution procedure from [Image Super-Resolution via Iterative Refinement](https://arxiv.org/abs/2104.07636)
is implemented as well.
A super-resolution model that upscales 256px chest X-rays to 1024px resolution using the MaCheX dataset is trained over `03_train_sr3_ddp.py`.
This model and conifguration reproduces the `CheffSR` model.
