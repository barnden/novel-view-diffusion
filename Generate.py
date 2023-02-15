import os
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from datasets.NMR import NMR
from XUNet import XUNet, logsnr_schedule_cosine


@torch.no_grad()
def get_noise_level(N, timestep, device="cpu"):
    '''
    N : batch dimension
    timestep : timestep normalised to [0, 1]
    device : device to create tensor on
    '''

    logsnr = timestep * torch.ones((N, 1), dtype=torch.float32, device=device)
    logsnr = logsnr_schedule_cosine(logsnr)

    mu = torch.sqrt(torch.sigmoid(logsnr))[..., None, None]
    sigma = torch.sqrt(torch.sigmoid(-logsnr))[..., None, None]

    return (logsnr, mu, sigma)

@torch.no_grad()
def synthesize(model, X, K, R, t, device=None):
    # TODO: Implement stochastic conditioning sampler
    # FIXME: This doesn't actually work ;(
    """
    Synthesize multiple views from single input.

    Parameters:
        X : dict {
                "image" : Tensor (C, W, H),
                "R"     : Tensor (3, 3),
                "t"     : Tensor (3)
            }
        K : Tensor (N, 3, 3)
        R : Tensor (N, 3, 3)
        t : Tensor (N, 3)

    Returns:
        Tensor (N, C, W, H)
    """
    device = device if device is not None else model.device

    N = K.shape[0]
    X = {k: v[None].repeat(N, *(v.ndim * (1,))) for (k, v) in X.items()}

    z = torch.randn_like(X["image"], device=device)

    logsnr, mu, sigma = get_noise_level(z.shape[0], 255/256, device)
    z = (mu * X["image"].to(device)) + (sigma * z)
    for i in reversed(range(256)):
        batch = {
            "x": X["image"],
            "z": z,
            "logsnr": logsnr,
            "K": K,
            "R": torch.stack([X["R"], R], dim=1),
            "t": torch.stack([X["t"], t], dim=1),
        }

        batch = {k: v.to(device) for (k, v) in batch.items()}
        E_theta = model(batch)

        x_hat = 1 / mu * (z - sigma * E_theta)
        logsnr, mu, sigma = get_noise_level(z.shape[0], (i - 1) / 256, device)
        E = torch.randn_like(x_hat)
        z = (mu * x_hat) + (sigma * E)

    return x_hat


if __name__ == "__main__":
    resolution = (64, 64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (2 * t) - 1),
        ]
    )
    dataset = NMR(root="./data", category="vessel", split="train")

    model = XUNet(resolution)
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load("NMR.pth"))


    #518
    X, K, R, t = dataset[1201]
    idx = 0

    save_image(X[idx], "./generated/condition.png")
    reference = {"image": X[idx], "R": R[idx], "t": t[idx]}

    K = torch.concat((K[:idx], K[idx + 1 :]), dim=0)[:, None].repeat(
        1, 2, *((K.ndim - 1) * (1,))
    )
    R = torch.concat((R[:idx], R[idx + 1 :]), dim=0)
    t = torch.concat((t[:idx], t[idx + 1 :]), dim=0)

    views = synthesize(model, reference, K, R, t, device=device)

    save_image(views, "./generated/generated.png")
