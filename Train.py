"""
Train XUNet with NMR dataset
"""

import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from datasets.NMR import NMR
from XUNet import XUNet


def logsnr_schedule_cosine(t, *, logsnr_min=-20.0, logsnr_max=20.0):
    b = math.atan(math.exp(-0.5 * logsnr_max))
    a = math.atan(math.exp(-0.5 * logsnr_min)) - b

    return -2.0 * torch.log(torch.tan(a * t + b))


def create_dataloader(split="train", batch_size=8, resolution=(128, 128), workers=24):
    transform = transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (2 * t) - 1),
        ]
    )

    dataset = NMR("./data", category="chair", split=split, transform=transform)

    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, pin_memory=True
    )

    return loader


@torch.no_grad()
def create_batch(data):
    # Each batch contains 24 synthesized views.
    # Our 'batch' then becomes (batch_size * 12).
    X, K, R, t = data
    V = X.shape[1]

    # Shuffle views
    perm = torch.randperm(V)
    X = X[:, perm]
    K = K[:, perm]
    R = R[:, perm]
    t = t[:, perm]

    # Split data into conditioning set (x) and noised set (z)
    batch = {
        "x": X[:, : V // 2],
        "z": X[:, V // 2 :],
        "K": torch.stack([K[:, : V // 2], K[:, V // 2 :]], dim=2),
        "R": torch.stack([K[:, : V // 2], K[:, V // 2 :]], dim=2),
        "t": torch.stack([t[:, : V // 2], t[:, V // 2 :]], dim=2),
    }

    batch = {k: v.flatten(0, 1).to(device) for (k, v) in batch.items()}

    # fmt: off
    # Generate noise levels
    step = (torch.randint(0, 256, size=(V // 2, 1), dtype=torch.float32, device=device) / 256)

    logsnr = logsnr_schedule_cosine(step)
    batch["logsnr"] = logsnr
    # fmt: on

    return batch


@torch.no_grad()
def create_noise(logsnr, x):
    E_actual = torch.randn_like(x)

    mu = torch.sqrt(torch.sigmoid(logsnr))[..., None, None]
    sigma = torch.sqrt(torch.sigmoid(-logsnr))[..., None, None]

    return (E_actual, mu, sigma)


@torch.no_grad()
def synthesize_images(model, loader):
    # FIXME: Validate against more than just one batch
    batch = create_batch(next(iter(loader)))
    E_actual, mu, sigma = create_noise(batch["logsnr"], batch["z"])

    batch["z"] = (mu * batch["z"]) + (sigma * E_actual)

    E_predicted = model(batch)

    loss = F.mse_loss(E_predicted, E_actual)

    recovered = 1 / mu * (batch["z"] - sigma * E_predicted)

    # Normalize images : [-1, 1] to [0, 1]
    recovered = (recovered + 1) / 2

    return (loss, recovered)


if __name__ == "__main__":
    resolution = (64, 64)
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = XUNet(resolution)
    model = nn.DataParallel(model)
    model.to(device)

    data_train = create_dataloader(split="train", batch_size=batch_size, resolution=resolution)
    data_validate = create_dataloader(split="val", batch_size=batch_size, resolution=resolution)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))

    if os.path.exists("NMR.pth"):
        model.load_state_dict(torch.load("NMR.pth"))

    for epoch in range(10):
        print("Epoch", epoch)

        for idx, data in enumerate(data_train):
            optimizer.zero_grad()

            batch = create_batch(data)

            # Add noise to frames
            E_actual, mu, sigma = create_noise(batch["logsnr"], batch["z"])

            batch["z"] = (mu * batch["z"]) + (sigma * E_actual)

            # Predict noise
            E_predicted = model(batch)

            # Compute loss
            loss = F.mse_loss(E_predicted, E_actual)

            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                # FIXME: Plot losses in addition to logging
                print()
                print(f"Epoch {epoch} Batch {idx}")
                print(f"\ttrain={loss}")
                print(f"\tvalid={validation_loss}")
                print()
                validation_loss, images = synthesize_images(model, data_validate)
                save_image(images, f"./images/validation_{epoch}_{idx}.png")

                if idx > 0:
                    state = model.state_dict()
                    torch.save(state, f"NMR.pth")

                    if idx % 500 == 0:
                        torch.save(state, f"NMR_{epoch}_{idx}")
