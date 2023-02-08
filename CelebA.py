"""
Train the XUNet model on regular images from CelebA dataset
without any pose information to test if the diffusion process
is working properly.
"""

import math
import os

import PIL

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.utils import save_image

from XUNet import XUNet


def logsnr_schedule_cosine(t, *, logsnr_min=-20.0, logsnr_max=20.0):
    b = math.atan(math.exp(-0.5 * logsnr_max))
    a = math.atan(math.exp(-0.5 * logsnr_min)) - b

    return -2.0 * torch.log(torch.tan(a * t + b))


def sigmoid(t):
    y = 1 + torch.exp(-t)

    return 1 / y


class CelebA(datasets.CelebA):
    def __getitem__(self, index):
        # Open image file
        X = PIL.Image.open(
            os.path.join(
                self.root, self.base_folder, "img_align_celeba", self.filename[index]
            )
        )

        # Apply transform, assume converted to torch tensor
        if self.transform is not None:
            X = self.transform(X)

        return {
            "x": X,
            "z": X,
            "t": torch.zeros((2, 3)),
            "R": torch.stack(2 * (torch.eye(3),), dim=0),
            "K": torch.stack(2 * (torch.eye(3),), dim=0),
        }


def create_dataloader(resolution=(128, 128), batch_size=128):
    transform = transforms.Compose(
        [
            transforms.Resize(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (2 * t) - 1),
        ]
    )

    data = CelebA("./data", split="train", transform=transform, download=False)

    loader = DataLoader(data, batch_size=batch_size, num_workers=24, pin_memory=True)

    return loader

@torch.no_grad()
def synthesize_and_save_images(model, x):
    # Step ~ Unif{0, 255}, normalised to [0, 1]
    N = x.shape[0]
    step = torch.randint(0, 256, size=(N, 1), dtype=torch.float32, device="cuda") / 256

    # Compute lambda
    logsnr = logsnr_schedule_cosine(step)

    # Generate noise
    z = torch.randn_like(x, device="cuda")

    batch = {
        "x": x,
        "z": z,
        "logsnr": logsnr,
        "t": torch.zeros((N, 2, 3), device="cuda"),
        "R": torch.stack(2 * (torch.eye(3, device="cuda"),), dim=0).repeat(N, 1, 1).reshape(N, 2, 3, 3),
        "K": torch.stack(2 * (torch.eye(3, device="cuda"),), dim=0).repeat(N, 1, 1).reshape(N, 2, 3, 3),
    }

    # Noise prediction
    E = model(batch)

    # Remove noise from z
    mu = torch.sqrt(sigmoid(logsnr))[..., None, None]
    sigma = torch.sqrt(sigmoid(-logsnr))[..., None, None]

    recovered = 1 / mu * (z - sigma * E)

    # Normalize [-1, 1] -> [0, 1]
    recovered = (recovered + 1) / 2

    save_image(recovered, f"./images/result_{epoch}_{batch_idx}.png")


if __name__ == "__main__":
    B = 8
    resolution = 32

    resolution = 2 * (resolution,)
    loader = create_dataloader(resolution=resolution, batch_size=B)

    model = XUNet(resolution, ch_mult=(1, 2, 4))
    model.to("cuda")

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99))

    if os.path.exists("celeba.pth"):
        model.load_state_dict(torch.load("celeba.pth"))

    print("Training XUNet on CelebA (no pose information)")
    for epoch in range(10):
        print("Epoch", epoch)

        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()
            N = batch['x'].shape[0]

            # Everything to CUDA!
            # - there's probably a better way to do this
            batch["x"] = batch["x"].to("cuda")
            batch["z"] = batch["z"].to("cuda")
            batch["t"] = batch["t"].to("cuda")
            batch["R"] = batch["R"].to("cuda")
            batch["K"] = batch["K"].to("cuda")

            # Step ~ Unif{0, 255}, normalised to [0, 1]
            step = torch.randint(0, 256, size=(N, 1), dtype=torch.float32, device="cuda") / 256

            # Compute lambda
            logsnr = logsnr_schedule_cosine(step)

            # Add logsnr to batch dict
            batch["logsnr"] = logsnr

            # Create Gaussian noise
            actual_noise = torch.randn_like(batch["z"]).to("cuda")

            # Compute shift/scale for Gaussian
            mu = torch.sqrt(sigmoid(logsnr))[..., None, None]
            sigma = torch.sqrt(sigmoid(-logsnr))[..., None, None]

            # Add noise to images
            z = batch["z"].detach().clone()
            batch["z"] = mu * batch["z"] + sigma * actual_noise

            # Model prediction
            predicted_noise = model(batch)

            # Compute loss
            loss = F.mse_loss(predicted_noise, actual_noise)

            # Backpropagate
            loss.backward()
            optimizer.step()

            if batch_idx % 200 == 0:
                print("loss", loss)

                synthesize_and_save_images(model, z)

                if batch_idx > 0:
                    torch.save(model.state_dict(), "celeba.pth")
