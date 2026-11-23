import os
import torch

from common.data import get_data
from common.train import evaluate
from common.utils import load_json, save_json
from Q1.q1a_train import MODELS, CONFIG, HERE


def main():
    data = get_data()
    summary = {}
    for name, spec in MODELS.items():
        runs = []
        for lr in CONFIG["lr_grid"]:
            path = os.path.join(HERE, "results", f"q1a_{name}_lr{lr}.json")
            if os.path.exists(path):
                runs.append(load_json(path))
        if not runs:
            continue
        sweep = {r["lr"]: r["history"]["val_acc"][-1] for r in runs}
        best = max(runs, key=lambda r: r["history"]["val_acc"][-1])
        h = best["history"]

        model = spec["build"]().cuda()
        model.load_state_dict(torch.load(os.path.join(HERE, "checkpoints", f"q1a_{name}_lr{best['lr']}.pt")))
        _, test_acc = evaluate(model, *data["test"])

        summary[name] = {"params": best["params"], "aug": best["aug"], "selected_lr": best["lr"],
                         "val_acc_by_lr": sweep, "train_acc": h["train_acc"][-1],
                         "val_acc": h["val_acc"][-1], "test_acc": test_acc,
                         "gap_train_val": h["train_acc"][-1] - h["val_acc"][-1]}
        print(f"{name:9s} params {best['params']:>9,d}  lr {best['lr']:<5} train {h['train_acc'][-1]:.4f}  "
              f"val {h['val_acc'][-1]:.4f}  test {test_acc:.4f}")
    save_json(summary, os.path.join(HERE, "results", "q1a_summary.json"))


if __name__ == "__main__":
    main()

