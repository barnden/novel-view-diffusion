import torch

@torch.no_grad()
def pad(tensor: torch.Tensor, ndim):
    while (tensor.ndim != ndim):
        tensor = tensor[..., None]

    return tensor

@torch.no_grad()
def pad_like(tensor: torch.Tensor, like: torch.Tensor):
    return pad(tensor, like.ndim)

@torch.no_grad()
def trim_end(tensor: torch.Tensor):
    while (tensor.shape[-1] == 1):
        tensor = tensor.squeeze(-1)

    return tensor
