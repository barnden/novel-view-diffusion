"""
Creates a custom dataset object for renderings of the ShapeNet dataset
using the Stanford Shapenet Renderer script

FIXME: Currently, this assumes all renders are in the base dir
       However, it would make more sense to have renders split
       by taxonomy.

FIXME: Assumption of default Stanford renderer options
       Currently: location=(0, 1, 0.6), lens=35deg, sensor_width=32

Retrieve intrinsic using get_calibration_matrix_K_from_blender()
by https://mcarletti.github.io/articles/blenderintrinsicparams/
"""

import os
from functools import partial
import json

import PIL
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

class ShapeNet(Dataset):
    def __init__(
        self, root, base="ShapeNet", split="train", views=12, transform=None
    ):
        self.root = root
        self.base = base
        self.views = views
        self.transform = transform

        if split == "val" or split == "validate":
            split = "validation"

        assert split in ["validation", "test", "train"]

        with open(os.path.join(root, base, f"{split}.lst")) as f:
            self.files = f.read().splitlines()

        # Load camera information
        # FIXME: Add ability for different camera intrinsics/poses for each model
        # FIXME: Don't use pickle?
        self.cameras = np.load(os.path.join(root, base, "cameras.npy"), allow_pickle=True).item()

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # This yields all views for the object
        assert index < len(self.files)

        path = partial(os.path.join, self.root, self.base, self.files[index])

        stepsize = 360 // self.views

        # Load rendered images
        load_view = lambda idx: PIL.Image.open(
            path(f"{self.files[index]}_r_{str(idx * stepsize).zfill(3)}.png")
        )

        X = [load_view(i) for i in range(self.views)]

        if self.transform is not None:
            # Assumption is that transform composition does ToTensor()
            X = [self.transform(x) for x in X]
        else:
            X = [transforms.ToTensor()(x) for x in X]

        X = torch.stack(X, dim=0)

        # Convert transparent background to white
        for i in range(3):
            X[:, i, :, :][X[:, -1, :, :] == 0] = 1

        X = X[:, :3, :, :]

        save_image(X, "asdasd.png")

        # Get camera intrinsics
        intrinsic = lambda idx: torch.Tensor(self.cameras[f'camera_mat_{idx}'])
        K = torch.stack([intrinsic(i) for i in range(self.views)])[:, :3, :3]

        # Get camera extrinsics and split into R and t matrices
        extrinsic = lambda idx: torch.Tensor(self.cameras[f'world_mat_{idx}'])
        E = torch.stack([extrinsic(i) for i in range(self.views)])

        R = E[:, :3, :3]
        t = E[:, :3, -1]

        return (X, K, R, t)
