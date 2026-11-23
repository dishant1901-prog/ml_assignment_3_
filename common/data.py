import os
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT, "data", "cifar10.npz")
SPLIT_FILE = os.path.join(ROOT, "common", "split_seed242003.npz")

SPLIT_SEED = 242003

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def load_cifar():
    d = np.load(DATA_FILE)
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def make_split():
    from sklearn.model_selection import train_test_split
    _, y, _, _ = load_cifar()
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=5000, stratify=y, random_state=SPLIT_SEED, shuffle=True)
    tr.sort()
    va.sort()
    np.savez(SPLIT_FILE, train_idx=tr, val_idx=va, seed=SPLIT_SEED)
    return tr, va


def load_split():
    if not os.path.exists(SPLIT_FILE):
        raise FileNotFoundError("split file missing, run: python -m common.make_split")
    s = np.load(SPLIT_FILE)
    return s["train_idx"], s["val_idx"]


def get_data(device="cuda"):
    x_all, y_all, x_test, y_test = load_cifar()
    tr, va = load_split()

    def to_t(x):
        return torch.tensor(x, dtype=torch.float32, device=device) / 255.0

    xtr, xva, xte = to_t(x_all[tr]), to_t(x_all[va]), to_t(x_test)
    mean = xtr.mean(dim=(0, 2, 3), keepdim=True)
    std = xtr.std(dim=(0, 2, 3), keepdim=True)
    norm = lambda x: (x - mean) / std

    def lab(y):
        return torch.tensor(y, dtype=torch.long, device=device)

    return {
        "train": (norm(xtr), lab(y_all[tr])),
        "val": (norm(xva), lab(y_all[va])),
        "test": (norm(xte), lab(y_test)),
        "mean": mean, "std": std,
    }


def batches(x, y, batch_size, shuffle=True, gen=None):
    n = x.shape[0]
    if shuffle:
        order = torch.randperm(n, device=x.device, generator=gen)
    else:
        order = torch.arange(n, device=x.device)
    for i in range(0, n, batch_size):
        j = order[i:i + batch_size]
        yield x[j], y[j]


def augment(x):
    n = x.shape[0]
    flip = torch.rand(n, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    xp = torch.nn.functional.pad(x, (4, 4, 4, 4))
    dx = torch.randint(0, 9, (n,), device=x.device)
    dy = torch.randint(0, 9, (n,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (dy[:, None] + ar[None, :])[:, None, :, None].expand(n, 3, 32, 40)
    cols = (dx[:, None] + ar[None, :])[:, None, None, :].expand(n, 3, 32, 32)
    return xp.gather(2, rows).gather(3, cols)

