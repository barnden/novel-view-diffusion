'''
Adaptation of Visu3d's pinhole camera to PyTorch for use in XUNet.
'''

import torch
import torch.nn.functional as F


class Camera:
    def __init__(self, H, W, K, R, t):
        self.padding = 6 - R.ndim
        self.device = K.device

        self.H = H
        self.W = W

        Kinv = torch.inverse(K)

        self.R = R.reshape(R.shape[:-2] + (1,) * self.padding + R.shape[-2:])
        self.t = t.reshape(t.shape[:-1] + (1,) * self.padding + t.shape[-1:])
        self.Kinv = Kinv.reshape(Kinv.shape[:-2] + (1,) * self.padding + Kinv.shape[-2:])

    def centers(self):
        h, w = torch.meshgrid(
            torch.arange(self.H, device=self.device),
            torch.arange(self.W, device=self.device),
            indexing="xy")

        points = torch.stack([h, w], axis=-1)
        points = torch.asarray(points) + 0.5

        points = torch.concat([points, torch.ones_like(h)[..., None]], dim=-1)

        return points

    def rays(self):
        target = self.centers()
        target = target.reshape((1,) * self.padding + target.shape + (1,))
        dirs = self.R @ (self.Kinv @ target)
        dirs = dirs.squeeze()
        dirs = F.normalize(dirs, dim=-1)

        pos = self.t.repeat(1, 1, self.H, self.W, 1)

        pos = pos.squeeze()
        dirs = dirs.squeeze()

        return (pos, dirs)
