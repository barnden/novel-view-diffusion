"""
Creates a custom dataset object for NMR ShapeNet renderings.
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


class NMR(Dataset):
    def __init__(
        self, root, base="NMR_Dataset", category="all", split="train", transform=None
    ):
        self.root = root
        self.base = base
        self.transform = transform

        if isinstance(category, str):
            category = [category]

        assert isinstance(category, list)

        with open(os.path.join(root, base, "metadata.yaml")) as metadata:
            metadata = json.load(metadata)

        if "all" in category:
            categories = list(metadata.keys())
        else:
            categories = [
                k for (k, v) in metadata.items() for c in category if c in v["name"]
            ]

        if split == "validate":
            split = "val"

        assert split in ["val", "test", "train"]

        files = []
        for cat in categories:
            with open(os.path.join(root, base, cat, f"{split}.lst")) as f:
                lines = f.read().splitlines()
                files += [os.path.join(cat, line) for line in lines]

        self.files = files

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # This yields all 24 views for the object
        assert index < len(self.files)

        path = partial(os.path.join, self.root, self.base, self.files[index])

        # Load rendered images
        load_view = lambda idx: PIL.Image.open(
            path("image", f"{str(idx).zfill(4)}.png")
        )

        X = [load_view(i) for i in range(24)]

        if self.transform is not None:
            # Assumption is that transform composition does ToTensor()
            X = [self.transform(x) for x in X]
        else:
            X = [transforms.ToTensor(x) for x in X]

        X = torch.stack(X, dim=0)

        # Load camera information
        cameras = np.load(path("cameras.npz"))

        # Get camera intrinsics
        intrinsic = lambda idx: torch.Tensor(cameras[f'camera_mat_{idx}'])
        K = torch.stack([intrinsic(i) for i in range(24)])[:, :3, :3]

        # Get camera extrinsics and split into R and t matrices
        extrinsic = lambda idx: torch.Tensor(cameras[f'world_mat_{idx}'])
        E = torch.stack([extrinsic(i) for i in range(24)])

        R = E[:, :3, :3]
        t = E[:, :3, -1]

        return (X, K, R, t)
