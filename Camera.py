"""
Adaptation of Visu3d's pinhole camera to PyTorch for use in XUNet.
"""

import torch
import torch.nn.functional as F

from math import cos, sin, radians

from Utils import trim_end


class Camera:
    def __init__(self, H, W, K, R, t):
        # Assume we have batch tensors of camera poses
        # Each batch contains:
        #   - 2 rotation and translation (condition/noisy poses)
        #   - 1 camera intrinsic matrix

        assert R.ndim == 4 and list(R.shape[1:]) == [2, 3, 3]
        assert K.ndim == 3 and list(K.shape[1:]) == [3, 3]
        assert t.ndim == 3 and list(t.shape[1:]) == [2, 3]

        self.device = K.device
        self.H = H
        self.W = W

        self.R = R  # B, 2, 3, 3
        self.t = t  # B, 2, 3
        self.Kinv = torch.inverse(K)  # B, 3, 3

    def centers(self):
        # Generate a 2D H x W grid of 3D image plane coordinates at z=1
        points = torch.meshgrid(
            torch.arange(self.H, device=self.device),
            torch.arange(self.W, device=self.device),
            indexing="xy",
        )

        points = torch.stack(points, axis=-1)
        points = torch.asarray(points) + 0.5

        points = torch.concat(
            [points, torch.ones_like(points[..., 0])[..., None]], dim=-1
        )

        return points

    def _rays(self, P1, P2):
        # Suppose that:
        #    P1 is the conditioning pose
        #    P2 is the noisy view (novel view to synthesize)
        # generate absolute NeRF-like rays of P1 then relativize them w.r.t. P2
        # see: Scene Representation Transformer by Sajjadi et al. (2021).

        # P1/P2 are dictionaries with "R" and "t" for the camera extrinsics

        # Generate absolute NeRF rays
        target = self.centers()
        dirs = (
            P1["R"][:, None, None]
            @ self.Kinv[:, None, None]
            @ target[None, ..., None]
        )
        pos = (
            P1["t"][:, None]
            .repeat(1, self.H * self.W, 1)
            .reshape(-1, self.H, self.W, 3)
        )

        # Create extrinsic [R | t] for noisy view
        # Want: B, 3, 3 (+) B, 3 -> B, 3, 4
        E = P2["R"].transpose(-1, -2)

        # FIXME: Is there a better way?
        E = (
            torch.hstack((E[:, 0], E[:, 1], E[:, 2], P2["t"]))
            .reshape(-1, 4, 3)
            .transpose(-1, -2)
        )
        E = torch.hstack(
            (
                E[:, 0],
                E[:, 1],
                E[:, 2],
                torch.tensor((0, 0, 0, 1), device=self.device, dtype=torch.float32)[None].repeat(E.shape[0], 1),
            )
        ).reshape(-1, 4, 4)

        E = E[:, None, None]

        # Relativize rays wrt noisy view
        Einv = torch.inverse(E)
        dirs = Einv[..., :3, :3] @ dirs

        pos = torch.concat((pos, torch.ones(pos.shape[:-1], device=self.device)[..., None]), dim=-1)
        pos = Einv @ pos[..., None]

        dirs = trim_end(dirs)

        # Drop homogeneous coordinate
        pos = trim_end(pos)[..., :-1]

        return (pos, dirs)

    def rays(self):
        # Create relativized NeRF-like camera rays

        # Outputs: Tuple (pos, dir)
        #   pos: Tensor ~ B, F, W, H, 3
        #   dir: Tensor ~ B, F, W, H, 3

        condition = {"R": self.R[:, 0], "t": self.t[:, 0]}
        noisy = {"R": self.R[:, 1], "t": self.t[:, 0]}

        x_pos, x_dir = self._rays(condition, noisy)
        z_pos, z_dir = self._rays(noisy, condition)

        pos = torch.stack((x_pos, z_pos), dim=1)
        dirs = torch.stack((x_dir, z_dir), dim=1)

        return (pos, dirs)
