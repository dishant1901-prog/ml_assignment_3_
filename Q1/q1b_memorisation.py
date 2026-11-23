import argparse
import copy
import os
import numpy as np
import torch

from common.data import get_data
from common.models import MLP
from common.train import train, evaluate
from common.utils import set_seed, count_params, save_json

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 242003

CONFIG = {
    "hidden": [2048, 2048],
    "epochs": 60,
    "evals_per_epoch": 8,
    "batch_size": 128,
    "optimizer": "SGD momentum 0.9 nesterov",
    "lr": 0.002,
    "schedule": "constant",
    "weight_decay": 0.0,
    "augmentation": False,
}


def corrupt_labels(y, frac, seed):
    rng = np.random.RandomState(seed)
    y = y.cpu().numpy().copy()
    n = len(y)
    idx = rng.choice(n, int(round(frac * n)), replace=False)
    shift = rng.randint(1, 10, size=len(idx))
    y_noisy = y.copy()
    y_noisy[idx] = (y[idx] + shift) % 10
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return y_noisy, mask


@torch.no_grad()
def predict_labels(model, x, bs=2000):
    model.eval()
    return torch.cat([model(x[i:i + bs]).argmax(1) for i in range(0, len(x), bs)])


def main(noise):
    data = get_data()
    xtr, ytr_true = data["train"]
    y_noisy, mask = corrupt_labels(ytr_true, noise, SEED)
    y_noisy = torch.tensor(y_noisy, device=xtr.device)
    mask_t = torch.tensor(mask, device=xtr.device)
    print(f"noise {noise}: {mask.sum()} relabelled, check all wrong:",
          bool((y_noisy[mask_t] != ytr_true[mask_t]).all().item()))

    train_data = {"train": (xtr, y_noisy), "val": data["val"]}
    set_seed(SEED)
    model = MLP(tuple(CONFIG["hidden"])).cuda()
    n_params = count_params(model)
    best = {"val_acc": -1.0, "epoch": 0, "state": None}

    def extra_eval(m, ep):
        pred = predict_labels(m, xtr)
        clean = ~mask_t
        out = {"clean_fit": (pred[clean] == y_noisy[clean]).float().mean().item()}
        if mask_t.any():
            out["noisy_fit"] = (pred[mask_t] == y_noisy[mask_t]).float().mean().item()
            out["noisy_true"] = (pred[mask_t] == ytr_true[mask_t]).float().mean().item()
        _, va = evaluate(m, *data["val"])
        if va > best["val_acc"]:
            best.update(val_acc=va, epoch=ep, state=copy.deepcopy(m.state_dict()))
        return out

    hist = train(model, train_data, epochs=CONFIG["epochs"], lr=CONFIG["lr"],
                 weight_decay=CONFIG["weight_decay"], batch_size=CONFIG["batch_size"],
                 schedule="const", seed=SEED, evals_per_epoch=CONFIG["evals_per_epoch"],
                 extra_eval=extra_eval)

    _, test_final = evaluate(model, *data["test"])
    model.load_state_dict(best["state"])
    _, test_best = evaluate(model, *data["test"])
    print(f"best val {best['val_acc']:.4f} at epoch {best['epoch']}  test(best-val ckpt) {test_best:.4f}  "
          f"test(final) {test_final:.4f}")

    tag = f"noise{int(noise * 100)}"
    save_json({"noise": noise, "n_relabelled": int(mask.sum()), "params": n_params, "config": CONFIG,
               "history": hist, "best_val_epoch": best["epoch"], "best_val_acc": best["val_acc"],
               "test_acc_best_val_ckpt": test_best, "test_acc_final": test_final},
              os.path.join(HERE, "results", f"q1b_{tag}.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, nargs="+", default=[0.2])
    args = ap.parse_args()
    for nz in args.noise:
        main(nz)

