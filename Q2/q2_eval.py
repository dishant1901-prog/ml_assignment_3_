import os
import numpy as np
import torch

from common.data import get_data
from common.train import predict
from common.utils import load_json, save_json
from Q1.q1a_train import MODELS, CONFIG
from Q2.transforms import apply_condition

HERE = os.path.dirname(os.path.abspath(__file__))
Q1_DIR = os.path.join(os.path.dirname(HERE), "Q1")
CONDITIONS = ["clean", "perm", "patch16", "patch8", "patch4"]


def run_files(cond, name, lr):
    if cond == "clean":
        return (os.path.join(Q1_DIR, "results", f"q1a_{name}_lr{lr}.json"),
                os.path.join(Q1_DIR, "checkpoints", f"q1a_{name}_lr{lr}.pt"))
    return (os.path.join(HERE, "results", f"q2_{cond}_{name}_lr{lr}.json"),
            os.path.join(HERE, "checkpoints", f"q2_{cond}_{name}_lr{lr}.pt"))


def select(cond, name):
    runs = []
    for lr in CONFIG["lr_grid"]:
        rj, ck = run_files(cond, name, lr)
        if os.path.exists(rj):
            runs.append((load_json(rj), ck))
    if not runs:
        return None
    return max(runs, key=lambda r: r[0]["history"]["val_acc"][-1])


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def analyse(probs, y):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = pred == y
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y, pred), 1)
    return {
        "test_acc": float(correct.mean()),
        "mean_conf": float(conf.mean()),
        "mean_conf_correct": float(conf[correct].mean()),
        "mean_conf_wrong": float(conf[~correct].mean()),
        "frac_wrong_with_conf_over_0.9": float(((conf > 0.9) & ~correct).sum() / max(1, (~correct).sum())),
        "ece": ece(conf, correct),
        "per_class_acc": (np.diag(cm) / cm.sum(1)).tolist(),
        "confusion": cm.tolist(),
    }


def main():
    base = get_data()
    y_test = base["test"][1].cpu().numpy()
    summary, probs_out = {}, {}
    for cond in CONDITIONS:
        data = apply_condition(base, cond)
        for name in ["mlp", "cnn"]:
            sel = select(cond, name)
            if sel is None:
                continue
            res, ck = sel
            model = MODELS[name]["build"]().cuda()
            model.load_state_dict(torch.load(ck))
            p = predict(model, data["test"][0]).cpu().numpy()
            a = analyse(p, y_test)
            a.update(selected_lr=res["lr"], params=res["params"], val_acc=res["history"]["val_acc"][-1],
                     train_acc=res["history"]["train_acc"][-1],
                     lrs_tried=sorted(set(r for r in CONFIG["lr_grid"]
                                          if os.path.exists(run_files(cond, name, r)[0]))))
            summary[f"{cond}/{name}"] = a
            probs_out[f"{cond}__{name}"] = p
            print(f"{cond:8s} {name:4s} lr {res['lr']:<5} val {a['val_acc']:.4f}  test {a['test_acc']:.4f}  "
                  f"conf ok/wrong {a['mean_conf_correct']:.3f}/{a['mean_conf_wrong']:.3f}  ECE {a['ece']:.3f}")

    for name in ["mlp", "cnn"]:
        sel = select("clean", name)
        if sel is None:
            continue
        model = MODELS[name]["build"]().cuda()
        model.load_state_dict(torch.load(sel[1]))
        for cond in ["perm", "patch16", "patch8", "patch4"]:
            p = predict(model, apply_condition(base, cond)["test"][0]).cpu().numpy()
            a = analyse(p, y_test)
            summary[f"cleantrained_on_{cond}/{name}"] = a
            probs_out[f"cleantrained_on_{cond}__{name}"] = p
            print(f"clean-trained {name} tested on {cond}: test {a['test_acc']:.4f}  "
                  f"conf {a['mean_conf']:.3f}")

    save_json(summary, os.path.join(HERE, "results", "q2_summary.json"))
    np.savez_compressed(os.path.join(HERE, "results", "q2_test_probs.npz"), y=y_test, **probs_out)


if __name__ == "__main__":
    main()

