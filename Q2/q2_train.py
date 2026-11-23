import argparse
import os
import torch

from common.data import get_data
from common.train import train
from common.utils import set_seed, count_params, save_json
from Q1.q1a_train import MODELS, CONFIG, SEED
from Q2.transforms import apply_condition

HERE = os.path.dirname(os.path.abspath(__file__))


def run(name, cond, data, lrs):
    spec = MODELS[name]
    assert not spec["aug"]
    for lr in lrs:
        tag = f"{cond}_{name}_lr{lr}"
        if os.path.exists(os.path.join(HERE, "results", f"q2_{tag}.json")):
            print(f"== {tag} already done, skipping")
            continue
        set_seed(SEED)
        model = spec["build"]().cuda()
        print(f"== {tag}  params={count_params(model)}", flush=True)
        hist = train(model, data, epochs=CONFIG["epochs"], lr=lr, weight_decay=CONFIG["weight_decay"],
                     batch_size=CONFIG["batch_size"], use_aug=False, seed=SEED)
        save_json({"condition": cond, "model": name, "lr": lr, "params": count_params(model),
                   "config": CONFIG, "history": hist},
                  os.path.join(HERE, "results", f"q2_{tag}.json"))
        os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(HERE, "checkpoints", f"q2_{tag}.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--model", nargs="+", default=["mlp", "cnn"])
    ap.add_argument("--lr", nargs="+", type=float, default=CONFIG["lr_grid"])
    ap.add_argument("--lr-from", default=None,
                    help="use the lr that had the best val accuracy for this model under another condition")
    args = ap.parse_args()
    if args.lr_from:
        from common.utils import load_json
        best = None
        for m in args.model:
            for lr in CONFIG["lr_grid"]:
                p = os.path.join(HERE, "results", f"q2_{args.lr_from}_{m}_lr{lr}.json")
                if os.path.exists(p):
                    v = load_json(p)["history"]["val_acc"][-1]
                    if best is None or v > best[1]:
                        best = (lr, v)
        args.lr = [best[0]]
        print(f"lr taken from {args.lr_from} (best val): {best[0]}")
    data = apply_condition(get_data(), args.condition)
    data.pop("test")
    for m in args.model:
        run(m, args.condition, data, args.lr)

