import math
import torch

# Functions taken from Watson et al. (2022), modified for PyTorch
def posenc_nerf(x, min_deg=0, max_deg=15):
    if min_deg == max_deg:
        return x

    device = x.device

    scales = torch.exp2(torch.arange(start=min_deg, end=max_deg, device=device))
    xb = torch.reshape(x[..., None, :] * scales[:, None], list(x.shape[:-1]) + [-1])
    emb = torch.sin(torch.concat([xb, xb + torch.pi / 2.0], axis=-1))

    return torch.concat([x, emb], axis=-1)


def posenc_ddpm(timesteps, emb_ch, max_time=1000):
    device = timesteps.device

    half_dim = emb_ch // 2
    timesteps *= 1000 / max_time

    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb)
    emb = emb.reshape(*([1] * (timesteps.ndim - 1)), emb.shape[-1])
    emb = timesteps[..., None] * emb
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], axis=-1)

    return emb
