import argparse
import os
import torch

from common.data import get_data
from common.models import MLP, CNN
from common.train import train
from common.utils import set_seed, count_params, save_json

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 242003

CONFIG = {
    "epochs": 40,
    "batch_size": 128,
    "optimizer": "SGD momentum 0.9 nesterov",
    "schedule": "OneCycle, 15% warmup",
    "weight_decay": 5e-4,
    "lr_grid": [0.01, 0.03, 0.1, 0.3, 1.0],
}

MODELS = {
    "mlp": dict(build=lambda: MLP((336, 336)), aug=False),
    "cnn": dict(build=lambda: CNN((64, 128, 256)), aug=False),
    "mlp_wide": dict(build=lambda: MLP((1152, 1152)), aug=False),
    "mlp_aug": dict(build=lambda: MLP((336, 336)), aug=True),
}


def run(name, data, lrs):
    spec = MODELS[name]
    for lr in lrs:
        if os.path.exists(os.path.join(HERE, "results", f"q1a_{name}_lr{lr}.json")):
            print(f"== {name}  lr={lr}  already done, skipping")
            continue
        set_seed(SEED)
        model = spec["build"]().cuda()
        n_params = count_params(model)
        print(f"== {name}  lr={lr}  params={n_params}", flush=True)
        hist = train(model, data, epochs=CONFIG["epochs"], lr=lr, weight_decay=CONFIG["weight_decay"],
                     batch_size=CONFIG["batch_size"], use_aug=spec["aug"], seed=SEED)
        tag = f"{name}_lr{lr}"
        save_json({"model": name, "lr": lr, "params": n_params, "aug": spec["aug"],
                   "config": CONFIG, "history": hist},
                  os.path.join(HERE, "results", f"q1a_{tag}.json"))
        os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(HERE, "checkpoints", f"q1a_{tag}.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=list(MODELS))
    ap.add_argument("--lr", nargs="+", type=float, default=CONFIG["lr_grid"])
    args = ap.parse_args()
    data = get_data()
    for m in args.model:
        run(m, data, args.lr)

