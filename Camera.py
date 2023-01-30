import torch
import torch.nn.functional as F

class Camera():
    def __init__(self, H, W, K, R, t):
        self.Kinv = torch.inverse(K)
        self.H = H
        self.W = W
        self.R = R
        self.t = t

    def centers(self):
        h, w = torch.meshgrid(torch.arange(self.H), torch.arange(self.W), indexing="xy")

        points = torch.stack([h, w], axis=-1)
        points = torch.asarray(points) + 0.5

        points = torch.concat([
            points,
            torch.ones_like(h)[..., None]
        ], dim=-1)

        return points

    def rays(self):
        target = self.centers().unsqueeze(-1)
        dirs = self.R @ (self.Kinv @ target)
        dirs = dirs.squeeze()
        dirs = F.normalize(dirs, dim=-1)

        pos = torch.stack(self.H * self.W * (self.t, ), dim=-1).reshape(self.H, self.W, 3)

        return (pos, dirs)
