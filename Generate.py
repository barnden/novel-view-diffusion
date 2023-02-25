import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

from datasets.NMR import NMR
from XUNet import XUNet, logsnr_schedule_cosine, get_schedule_constants

@torch.no_grad()
def get_noise_level(N, timestep, device="cpu"):
    """
    N : batch dimension
    timestep : timestep normalised to [0, 1]
    device : device to create tensor on
    """

    # FIXME: Squeeze logsnr
    logsnr = timestep * torch.ones((N, 1), dtype=torch.float32, device=device)
    logsnr = logsnr_schedule_cosine(logsnr)

    plogsnr  = torch.sqrt(torch.sigmoid(logsnr))[..., None, None]
    nlogsnr = torch.sqrt(torch.sigmoid(-logsnr))[..., None, None]

    return (logsnr, plogsnr, nlogsnr)



@torch.no_grad()
def get_variance(N, timestep, device="cpu"):
    a, b = get_schedule_constants(timestep)

    x = timestep * torch.ones((N,), dtype=torch.float32, device=device)
    x = a * x + b
    
    alpha = torch.cos(x)
    sigma = torch.sin(x)

    return (alpha, torch.square(sigma))


@torch.no_grad()
def pad(tensor: torch.Tensor, ndim):
    while (tensor.ndim != ndim):
        tensor = tensor[..., None]

    return tensor

@torch.no_grad()
def pad_like(tensor: torch.Tensor, like: torch.Tensor):
    return pad(tensor, like.ndim)


@torch.no_grad()
def synthesize(model, X, K, R, t, gamma=0., device=None):
    # TODO: Implement stochastic conditioning sampler
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
        gamma: Hyperparameter controlling amount of noise added during sampling
               From "Improved denoising diffusion probabilistic models", Nichol & Dhariwal, 2021.

    Returns:
        Tensor (N, C, W, H)
    """
    device = device if device is not None else model.device

    N = K.shape[0]
    X = {k: v[None].repeat(N, *(v.ndim * (1,))) for (k, v) in X.items()}

    logsnr_t, plogsnr_t, nlogsnr_t = get_noise_level(N, 255 / 256, device=device)
    alpha_t, sigma2_t = get_variance(N, 255 / 256, device=device)
    z = torch.randn_like(X["image"], device=device)

    for i in reversed(range(1, 255)):
        logsnr_s, plogsnr_s, nlogsnr_s = get_noise_level(N, i / 256, device=device)
        alpha_s, sigma2_s = get_variance(N, i / 256, device=device)

        batch = {
            "x": X["image"],
            "z": z,
            "logsnr": logsnr_t,
            "K": K,
            "R": torch.stack([X["R"], R], dim=1),
            "t": torch.stack([X["t"], t], dim=1),
        }

        batch = {k: v.to(device) for (k, v) in batch.items()}
        E_theta = model(batch)

        # See "Progressive Distillation for Fast Sampling of Diffusion Models" by Salimans and Ho, 2022.
        e_st = pad_like(torch.exp(logsnr_t - logsnr_s), z)
        e_ts = pad_like(torch.exp(logsnr_s - logsnr_t), z)
        sigma2_st = (1 - e_st) * pad_like(sigma2_s, z)
        sigma2_ts = (1 - e_ts) * pad_like(sigma2_t, z)

        x_hat = 1 / plogsnr_t * (z - nlogsnr_t * E_theta)

        # Clip to [-1, 1] per Watson et al. 2022 section 7.3
        x_hat = torch.clip(x_hat, -1, 1)

        C1 = e_st * pad_like((alpha_s / alpha_t), z)
        C2 = (1 - e_st) * pad_like(alpha_s, z)
        mu_tilde = C1 * z + C2 * x_hat
        sigma = torch.sqrt(sigma2_st ** (1 - gamma) * (sigma2_ts) ** gamma)

        E = torch.randn_like(z)

        z = mu_tilde + sigma * E

        logsnr_t, plogsnr_t, nlogsnr_t = logsnr_s, plogsnr_s, nlogsnr_s
        alpha_t, sigma2_t = alpha_s, sigma2_s

        if i % 10 == 0:
            save_image((z + 1) / 2, f"./generated/z{i}.png")

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
    # TODO: Use test split
    dataset = NMR(root="./data", category="vessel", split="train")

    model = XUNet(resolution)
    model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(torch.load("NMR.pth"))

    # TODO: Don't hardcode these
    X, K, R, t = dataset[183]
    idx = 4

    save_image(X[idx], "./generated/condition.png")
    reference = {"image": X[idx], "R": R[idx], "t": t[idx]}

    K = torch.concat((K[:idx], K[idx + 1 :]), dim=0)
    K = K[:, None].repeat(1, 2, *((K.ndim - 1) * (1,)))
    R = torch.concat((R[:idx], R[idx + 1 :]), dim=0)
    t = torch.concat((t[:idx], t[idx + 1 :]), dim=0)

    views = synthesize(model, reference, K, R, t, device=device)

    save_image(views, "./generated/generated.png")
