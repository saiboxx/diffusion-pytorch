"""Miscellaneous classes and functions."""
from typing import Any, Optional

import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.transforms import (
    Compose,
    Lambda,
    ToPILImage,
)
from torchvision.utils import make_grid


def transform_tensor_to_img() -> Compose:
    """Transform a tensor with a single element to a PIL image."""
    return Compose(
        [
            Lambda(lambda t: t.detach().cpu()),
            Lambda(lambda t: (t + 1) / 2),
            Lambda(lambda t: t.permute(1, 2, 0)),
            Lambda(lambda t: t * 255.0),
            Lambda(lambda t: t.numpy().astype(np.uint8)),
            ToPILImage(),
        ]
    )


def plot_image(
    img: Tensor,
    fig_size: Any = None,
    ncols: Optional[int] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> None:
    """Plot a tensor containing image data."""
    img = img.detach().cpu()

    # Shape of 4 implies multiple image inputs
    if len(img.shape) == 4:
        img = make_grid(img, nrow=ncols if ncols is not None else len(img))

    plt.figure(figsize=fig_size)
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()
    plt.close()
