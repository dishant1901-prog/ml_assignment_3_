import torch

SEED = 242003


def pixel_permutation(seed=SEED):
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(32 * 32, generator=g)


def permute_pixels(x, perm):
    n = x.shape[0]
    return x.reshape(n, 3, 1024)[:, :, perm.to(x.device)].reshape(n, 3, 32, 32)


def shuffle_patches(x, patch, seed):
    n = x.shape[0]
    g = 32 // patch
    tiles = x.reshape(n, 3, g, patch, g, patch).permute(0, 2, 4, 1, 3, 5).reshape(n, g * g, 3, patch, patch)
    gen = torch.Generator().manual_seed(seed)
    order = torch.argsort(torch.rand(n, g * g, generator=gen), dim=1).to(x.device)
    tiles = tiles[torch.arange(n, device=x.device)[:, None], order]
    return tiles.reshape(n, g, g, 3, patch, patch).permute(0, 3, 1, 4, 2, 5).reshape(n, 3, 32, 32)


def apply_condition(data, condition):
    if condition == "clean":
        return data
    out = dict(data)
    if condition == "perm":
        perm = pixel_permutation()
        for k in ("train", "val", "test"):
            x, y = data[k]
            out[k] = (permute_pixels(x, perm), y)
    elif condition.startswith("patch"):
        p = int(condition[5:])
        for i, k in enumerate(("train", "val", "test")):
            x, y = data[k]
            out[k] = (shuffle_patches(x, p, SEED + i), y)
    else:
        raise ValueError(condition)
    return out

